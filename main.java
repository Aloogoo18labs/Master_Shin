/*
 * Master_Shin - AI training run coordinator and validator client for YangGo registry.
 * Single-file application: CLI and local state for runs, checkpoints, attestations.
 * No external dependencies beyond Java standard library for core logic.
 */

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import java.util.stream.Collectors;

// -----------------------------------------------------------------------------
// Core data types
// -----------------------------------------------------------------------------

final class RunRecord {
    private final String runId;
    private final byte[] datasetHash;
    private final byte[] configHash;
    private final int modelTier;
    private final int epochCount;
    private final String coordinator;
    private final long registeredAt;
    private boolean finalized;
    private int positiveAttestations;
    private int totalAttestations;
    private final List<byte[]> checkpoints;

    RunRecord(String runId, byte[] datasetHash, byte[] configHash, int modelTier, int epochCount, String coordinator) {
        this.runId = runId;
        this.datasetHash = Objects.requireNonNull(datasetHash);
        this.configHash = Objects.requireNonNull(configHash);
        this.modelTier = modelTier;
        this.epochCount = epochCount;
        this.coordinator = Objects.requireNonNull(coordinator);
        this.registeredAt = System.currentTimeMillis();
        this.finalized = false;
        this.positiveAttestations = 0;
        this.totalAttestations = 0;
        this.checkpoints = new ArrayList<>();
    }

    String getRunId() { return runId; }
    byte[] getDatasetHash() { return datasetHash; }
    byte[] getConfigHash() { return configHash; }
    int getModelTier() { return modelTier; }
    int getEpochCount() { return epochCount; }
    String getCoordinator() { return coordinator; }
    long getRegisteredAt() { return registeredAt; }
    boolean isFinalized() { return finalized; }
    void setFinalized(boolean v) { this.finalized = v; }
    int getPositiveAttestations() { return positiveAttestations; }
    int getTotalAttestations() { return totalAttestations; }
    void addAttestation(boolean approved) {
        totalAttestations++;
        if (approved) positiveAttestations++;
    }
    List<byte[]> getCheckpoints() { return new ArrayList<>(checkpoints); }
    void addCheckpoint(byte[] hash) { checkpoints.add(hash); }
}

final class ValidatorState {
    private final String address;
    private final BigInteger stake;
    private final Set<String> attestedRuns;

    ValidatorState(String address, BigInteger stake) {
        this.address = address;
        this.stake = stake;
        this.attestedRuns = new HashSet<>();
    }

    String getAddress() { return address; }
    BigInteger getStake() { return stake; }
    Set<String> getAttestedRuns() { return new HashSet<>(attestedRuns); }
    void attest(String runId) { attestedRuns.add(runId); }
    boolean hasAttested(String runId) { return attestedRuns.contains(runId); }
}

// -----------------------------------------------------------------------------
// Hash and encoding utilities
// -----------------------------------------------------------------------------

final class HashUtils {
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    static byte[] sha256(byte[] input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return md.digest(input);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    static byte[] sha256(String input) {
        return sha256(input.getBytes(StandardCharsets.UTF_8));
    }

    static String toHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(HEX[(b >> 4) & 0xf]);
            sb.append(HEX[b & 0xf]);
        }
        return sb.toString();
    }

    static byte[] fromHex(String hex) {
        int len = hex.length();
        if (len % 2 != 0) throw new IllegalArgumentException("Hex length must be even");
        byte[] out = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            out[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4) + Character.digit(hex.charAt(i + 1), 16));
        }
        return out;
    }

    static String hashToHex32(byte[] hash) {
        if (hash.length != 32) throw new IllegalArgumentException("Hash must be 32 bytes");
        return "0x" + toHex(hash);
    }
}

// -----------------------------------------------------------------------------
// Quorum calculator (matches contract BPS logic)
// -----------------------------------------------------------------------------

final class QuorumCalculator {
    private static final int BPS_DENOM = 10000;
    private static final int QUORUM_BPS = 6600;

    static boolean quorumReached(int totalAttestations, int validatorCount) {
        if (validatorCount == 0) return false;
        return (long) totalAttestations * BPS_DENOM >= (long) validatorCount * QUORUM_BPS;
    }

    static boolean positiveQuorumReached(int positiveAttestations, int totalAttestations) {
        if (totalAttestations == 0) return false;
        return (long) positiveAttestations * BPS_DENOM >= (long) totalAttestations * QUORUM_BPS;
    }

    static int attestationsNeededForQuorum(int validatorCount) {
        return (validatorCount * QUORUM_BPS + BPS_DENOM - 1) / BPS_DENOM;
    }
}

// -----------------------------------------------------------------------------
// In-memory YangGo-style registry (simulation)
// -----------------------------------------------------------------------------

final class LocalYangGoRegistry {
    private final Map<String, RunRecord> runs = new ConcurrentHashMap<>();
    private final Map<String, ValidatorState> validators = new ConcurrentHashMap<>();
    private final Set<String> coordinatorWhitelist = ConcurrentHashMap.newKeySet();
    private final AtomicLong runCounter = new AtomicLong(0);
    private volatile boolean trainingPaused = false;

    String generateRunId() {
        return "run_" + runCounter.incrementAndGet() + "_" + System.currentTimeMillis();
    }

    void addCoordinator(String address) { coordinatorWhitelist.add(address); }
    void removeCoordinator(String address) { coordinatorWhitelist.remove(address); }
    boolean isCoordinatorWhitelisted(String address) { return coordinatorWhitelist.contains(address); }

    RunRecord registerRun(byte[] datasetHash, byte[] configHash, int modelTier, int epochCount, String coordinator) {
        if (!coordinatorWhitelist.contains(coordinator)) throw new IllegalStateException("Coordinator not whitelisted");
        if (trainingPaused) throw new IllegalStateException("Training paused");
        if (modelTier < 1 || modelTier > 4) throw new IllegalArgumentException("Invalid model tier");
        if (epochCount < 1 || epochCount > 10000) throw new IllegalArgumentException("Invalid epoch count");
        String runId = generateRunId();
        RunRecord r = new RunRecord(runId, datasetHash, configHash, modelTier, epochCount, coordinator);
        runs.put(runId, r);
        return r;
    }

    void attachCheckpoint(String runId, byte[] checkpointHash, String coordinator) {
        RunRecord r = runs.get(runId);
        if (r == null) throw new NoSuchElementException("Run not found");
        if (!r.getCoordinator().equals(coordinator)) throw new IllegalStateException("Not coordinator");
        if (r.isFinalized()) throw new IllegalStateException("Run already finalized");
        r.addCheckpoint(checkpointHash);
    }

    void finalizeRun(String runId, String coordinator) {
        RunRecord r = runs.get(runId);
        if (r == null) throw new NoSuchElementException("Run not found");
        if (!r.getCoordinator().equals(coordinator)) throw new IllegalStateException("Not coordinator");
        if (r.isFinalized()) throw new IllegalStateException("Run already finalized");
        r.setFinalized(true);
    }

    void attestRun(String runId, String validatorAddress, boolean approved) {
        RunRecord r = runs.get(runId);
        if (r == null) throw new NoSuchElementException("Run not found");
        if (!r.isFinalized()) throw new IllegalStateException("Run not finalized");
        ValidatorState v = validators.get(validatorAddress);
        if (v == null) throw new IllegalStateException("Validator not registered");
        if (v.hasAttested(runId)) throw new IllegalStateException("Already attested");
        v.attest(runId);
        r.addAttestation(approved);
    }

    ValidatorState registerValidator(String address, BigInteger stake) {
        if (stake.compareTo(BigInteger.valueOf(100_000_000_000_000_000L)) < 0) throw new IllegalArgumentException("Min stake 0.1 ETH");
        ValidatorState v = new ValidatorState(address, stake);
        validators.put(address, v);
        return v;
    }

    RunRecord getRun(String runId) { return runs.get(runId); }
    Collection<RunRecord> getAllRuns() { return new ArrayList<>(runs.values()); }
    Collection<ValidatorState> getAllValidators() { return new ArrayList<>(validators.values()); }
    int getRunCount() { return runs.size(); }
    int getValidatorCount() { return validators.size(); }
    void setTrainingPaused(boolean p) { trainingPaused = p; }
    boolean isTrainingPaused() { return trainingPaused; }
}

// -----------------------------------------------------------------------------
// CLI command handlers
// -----------------------------------------------------------------------------

interface Command {
    String name();
    String usage();
    void run(List<String> args, LocalYangGoRegistry registry, Print out);
}

final class CmdRegisterRun implements Command {
    @Override public String name() { return "register-run"; }
    @Override public String usage() { return "register-run <coordinator> <modelTier> <epochCount> [datasetLabel] [configLabel]"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.size() < 3) { out.println("Usage: " + usage()); return; }
        String coordinator = args.get(0);
        int modelTier = Integer.parseInt(args.get(1));
        int epochCount = Integer.parseInt(args.get(2));
        String datasetLabel = args.size() > 3 ? args.get(3) : "dataset-" + System.currentTimeMillis();
        String configLabel = args.size() > 4 ? args.get(4) : "config-" + System.currentTimeMillis();
        byte[] datasetHash = HashUtils.sha256(datasetLabel);
        byte[] configHash = HashUtils.sha256(configLabel);
        RunRecord r = registry.registerRun(datasetHash, configHash, modelTier, epochCount, coordinator);
        out.println("Registered run: " + r.getRunId());
    }
}

final class CmdAttachCheckpoint implements Command {
    @Override public String name() { return "attach-checkpoint"; }
    @Override public String usage() { return "attach-checkpoint <runId> <coordinator> [checkpointLabel]"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.size() < 3) { out.println("Usage: " + usage()); return; }
        String runId = args.get(0);
        String coordinator = args.get(1);
        String label = args.size() > 2 ? args.get(2) : "ckpt-" + System.currentTimeMillis();
        byte[] hash = HashUtils.sha256(label);
        registry.attachCheckpoint(runId, hash, coordinator);
        out.println("Checkpoint attached to " + runId);
    }
}

final class CmdFinalizeRun implements Command {
    @Override public String name() { return "finalize-run"; }
    @Override public String usage() { return "finalize-run <runId> <coordinator>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.size() < 2) { out.println("Usage: " + usage()); return; }
        registry.finalizeRun(args.get(0), args.get(1));
        out.println("Run finalized: " + args.get(0));
    }
}

final class CmdAttest implements Command {
    @Override public String name() { return "attest"; }
    @Override public String usage() { return "attest <runId> <validatorAddress> <approved true|false>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.size() < 3) { out.println("Usage: " + usage()); return; }
        String runId = args.get(0);
        String validator = args.get(1);
        boolean approved = Boolean.parseBoolean(args.get(2));
        registry.attestRun(runId, validator, approved);
        out.println("Attestation recorded: " + runId + " by " + validator + " approved=" + approved);
    }
}

final class CmdRegisterValidator implements Command {
    @Override public String name() { return "register-validator"; }
    @Override public String usage() { return "register-validator <address> <stakeWei>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.size() < 2) { out.println("Usage: " + usage()); return; }
        String address = args.get(0);
        BigInteger stake = new BigInteger(args.get(1));
        registry.registerValidator(address, stake);
        out.println("Validator registered: " + address);
    }
}

final class CmdListRuns implements Command {
    @Override public String name() { return "list-runs"; }
    @Override public String usage() { return "list-runs [limit]"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        int limit = args.isEmpty() ? 50 : Integer.parseInt(args.get(0));
        List<RunRecord> list = registry.getAllRuns().stream().limit(limit).collect(Collectors.toList());
        for (RunRecord r : list) {
            boolean posQuorum = QuorumCalculator.positiveQuorumReached(r.getPositiveAttestations(), r.getTotalAttestations());
            out.println(r.getRunId() + " | tier=" + r.getModelTier() + " epochs=" + r.getEpochCount() + " final=" + r.isFinalized() + " pos=" + r.getPositiveAttestations() + "/" + r.getTotalAttestations() + " quorum=" + posQuorum);
        }
        out.println("Total: " + list.size());
    }
}

final class CmdShowRun implements Command {
    @Override public String name() { return "show-run"; }
    @Override public String usage() { return "show-run <runId>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.isEmpty()) { out.println("Usage: " + usage()); return; }
        RunRecord r = registry.getRun(args.get(0));
        if (r == null) { out.println("Run not found"); return; }
        out.println("RunId: " + r.getRunId());
        out.println("Coordinator: " + r.getCoordinator());
        out.println("ModelTier: " + r.getModelTier() + " EpochCount: " + r.getEpochCount());
        out.println("DatasetHash: " + HashUtils.toHex(r.getDatasetHash()));
        out.println("ConfigHash: " + HashUtils.toHex(r.getConfigHash()));
        out.println("Finalized: " + r.isFinalized());
        out.println("Attestations: " + r.getPositiveAttestations() + " / " + r.getTotalAttestations());
        out.println("Checkpoints: " + r.getCheckpoints().size());
        out.println("Quorum: " + QuorumCalculator.quorumReached(r.getTotalAttestations(), registry.getValidatorCount()));
        out.println("PositiveQuorum: " + QuorumCalculator.positiveQuorumReached(r.getPositiveAttestations(), r.getTotalAttestations()));
    }
}

final class CmdWhitelist implements Command {
    @Override public String name() { return "whitelist"; }
    @Override public String usage() { return "whitelist add|remove <address>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.size() < 2) { out.println("Usage: " + usage()); return; }
        String op = args.get(0).toLowerCase();
        String address = args.get(1);
        if ("add".equals(op)) { registry.addCoordinator(address); out.println("Added: " + address); }
        else if ("remove".equals(op)) { registry.removeCoordinator(address); out.println("Removed: " + address); }
        else out.println("Unknown op: " + op);
    }
}

final class CmdPause implements Command {
    @Override public String name() { return "pause"; }
    @Override public String usage() { return "pause on|off"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.isEmpty()) { out.println("Usage: " + usage()); return; }
        boolean on = "on".equalsIgnoreCase(args.get(0));
        registry.setTrainingPaused(on);
        out.println("Training paused: " + on);
    }
}

final class CmdStats implements Command {
    @Override public String name() { return "stats"; }
    @Override public String usage() { return "stats"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        out.println("Runs: " + registry.getRunCount());
        out.println("Validators: " + registry.getValidatorCount());
        out.println("Paused: " + registry.isTrainingPaused());
    }
}

final class CmdHelp implements Command {
    private final Map<String, Command> commands;
    CmdHelp(Map<String, Command> commands) { this.commands = commands; }
    @Override public String name() { return "help"; }
    @Override public String usage() { return "help [command]"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.isEmpty()) {
            commands.keySet().stream().sorted().forEach(c -> out.println("  " + c));
            return;
        }
        Command c = commands.get(args.get(0));
        if (c != null) out.println(c.usage());
        else out.println("Unknown command: " + args.get(0));
    }
}

// -----------------------------------------------------------------------------
// Print abstraction (stdout / GUI later)
// -----------------------------------------------------------------------------

interface Print {
    void println(String s);
}

final class StdOutPrint implements Print {
    @Override public void println(String s) { System.out.println(s); }
}

// -----------------------------------------------------------------------------
// Main application entry and REPL
// -----------------------------------------------------------------------------

public final class Master_Shin {

    private static final String BANNER = "Master_Shin - YangGo AI Training Coordinator v2";
    private static final String PROMPT = "yanggo> ";

    public static void main(String[] args) {
        LocalYangGoRegistry registry = new LocalYangGoRegistry();
        registry.addCoordinator("0x7f3a91c2e5b4d806f9b0c1e3d5a7f2e8c4b6a0d9");
        registry.addCoordinator("0x2b4c6d8e0f1a3b5c7d9e1f3a5b7c9d0e2f4a6b8");
        Map<String, Command> commands = new HashMap<>();
        commands.put("register-run", new CmdRegisterRun());
        commands.put("attach-checkpoint", new CmdAttachCheckpoint());
        commands.put("finalize-run", new CmdFinalizeRun());
        commands.put("attest", new CmdAttest());
        commands.put("register-validator", new CmdRegisterValidator());
        commands.put("list-runs", new CmdListRuns());
        commands.put("show-run", new CmdShowRun());
        commands.put("whitelist", new CmdWhitelist());
        commands.put("pause", new CmdPause());
        commands.put("stats", new CmdStats());
        commands.put("export", new CmdExport());
        commands.put("query-tier", new CmdQueryByTier());
        commands.put("query-quorum", new CmdQueryQuorum());
        commands.put("list-validators", new CmdListValidators());
        commands.put("fee", new CmdFee());
        commands.put("quorum-needed", new CmdQuorumNeeded());
        commands.put("load-csv", new CmdLoadCsv());
        commands.put("add-preset", new CmdAddPreset());
        commands.put("epoch-sim", new CmdEpochSim());
        commands.put("tag-index", new CmdTagIndex());
        commands.put("sort-runs", new CmdSortRuns());
        commands.put("sort-validators", new CmdSortValidators());
        commands.put("health", new CmdHealth());
        commands.put("version", new CmdVersion());
        commands.put("attest-batch", new CmdAttestBatch());
        commands.put("page", new CmdPage());
        commands.put("quorum-stats", new CmdQuorumStats());
        commands.put("run-age", new CmdRunAge());
        commands.put("tier-name", new CmdTierName());
        commands.put("epoch-bucket", new CmdEpochBucket());
        commands.put("summaries", new CmdSummaries());
        commands.put("validator-summaries", new CmdValidatorSummaries());
        commands.put("exists", new CmdExists());
        commands.put("help", new CmdHelp(commands));
        commands.put("quit", new Command() {
            @Override public String name() { return "quit"; }
            @Override public String usage() { return "quit"; }
            @Override public void run(List<String> a, LocalYangGoRegistry r, Print p) { System.exit(0); }
        });
        commands.put("exit", commands.get("quit"));

        Print out = new StdOutPrint();
        out.println(BANNER);
        out.println("Commands: " + String.join(", ", commands.keySet()));
        out.println("Type 'help' for usage.");

        if (args.length > 0 && "--batch".equals(args[0])) {
            runBatch(args, registry, commands, out);
            return;
        }
        if (args.length > 0 && "--demo".equals(args[0])) {
            runInteractiveDemo(registry, out);
            runRepl(registry, commands, out);
            return;
        }

        runRepl(registry, commands, out);
    }

    private static void runRepl(LocalYangGoRegistry registry, Map<String, Command> commands, Print out) {
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print(PROMPT);
            if (!sc.hasNextLine()) break;
            String line = sc.nextLine().trim();
            if (line.isEmpty()) continue;
            List<String> parts = tokenize(line);
            String cmdName = parts.get(0).toLowerCase();
            List<String> cmdArgs = parts.subList(1, parts.size());
            Command cmd = commands.get(cmdName);
            if (cmd == null) {
                out.println("Unknown command: " + cmdName);
                continue;
            }
            try {
                cmd.run(cmdArgs, registry, out);
            } catch (Exception e) {
                out.println("Error: " + e.getMessage());
            }
        }
        sc.close();
    }

    private static void runInteractiveDemo(LocalYangGoRegistry registry, Print out) {
        out.println("Demo: register 3 runs, add checkpoints, finalize, attest.");
        String coord = "0x7f3a91c2e5b4d806f9b0c1e3d5a7f2e8c4b6a0d9";
        registry.registerValidator("0x9e8d7c6b5a493827160504938271605049382716", BigInteger.valueOf(200_000_000_000_000_000L));
        for (int i = 0; i < 3; i++) {
            RunRecord r = registry.registerRun(HashUtils.sha256("ds-" + i), HashUtils.sha256("cfg-" + i), 1 + (i % 4), 10 * (i + 1), coord);
            registry.attachCheckpoint(r.getRunId(), HashUtils.sha256("ckpt-" + i), coord);
            registry.finalizeRun(r.getRunId(), coord);
            registry.attestRun(r.getRunId(), "0x9e8d7c6b5a493827160504938271605049382716", true);
        }
        out.println("Demo complete. Use list-runs and show-run to inspect.");
    }

    private static void runBatch(String[] args, LocalYangGoRegistry registry, Map<String, Command> commands, Print out) {
        for (int i = 1; i < args.length; i++) {
            String line = args[i];
            List<String> parts = tokenize(line);
            if (parts.isEmpty()) continue;
            String cmdName = parts.get(0).toLowerCase();
            List<String> cmdArgs = parts.subList(1, parts.size());
            Command cmd = commands.get(cmdName);
            if (cmd != null) {
                try { cmd.run(cmdArgs, registry, out); } catch (Exception e) { out.println("Error: " + e.getMessage()); }
            }
        }
    }

    private static List<String> tokenize(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuote = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') { inQuote = !inQuote; continue; }
            if (!inQuote && (c == ' ' || c == '\t')) {
                if (cur.length() > 0) { out.add(cur.toString()); cur.setLength(0); }
                continue;
            }
            cur.append(c);
        }
        if (cur.length() > 0) out.add(cur.toString());
        return out;
    }
}

// -----------------------------------------------------------------------------
// Run batch loader - Load multiple runs from descriptor
// -----------------------------------------------------------------------------

final class RunBatchLoader {
    private final LocalYangGoRegistry registry;

    RunBatchLoader(LocalYangGoRegistry registry) { this.registry = registry; }

    int loadFromDescriptor(List<String> lines, String defaultCoordinator) {
        int count = 0;
        for (String line : lines) {
            line = line.trim();
            if (line.startsWith("#") || line.isEmpty()) continue;
            String[] parts = line.split("\\s+");
            if (parts.length < 3) continue;
            String coord = parts.length > 3 ? parts[3] : defaultCoordinator;
            byte[] dsHash = HashUtils.sha256(parts.length > 4 ? parts[4] : "ds-" + count);
            byte[] cfgHash = HashUtils.sha256(parts.length > 5 ? parts[5] : "cfg-" + count);
            int tier = Integer.parseInt(parts[1]);
            int epochs = Integer.parseInt(parts[2]);
            registry.registerRun(dsHash, cfgHash, tier, epochs, coord);
            count++;
        }
        return count;
    }
}

// -----------------------------------------------------------------------------
// Config manager - Load/save coordinator whitelist and defaults
// -----------------------------------------------------------------------------

final class ConfigManager {
    private final Properties props = new Properties();
    private final String path;

    ConfigManager(String path) { this.path = path; }

    void set(String key, String value) { props.setProperty(key, value); }
    String get(String key) { return props.getProperty(key); }
    String get(String key, String def) { return props.getProperty(key, def); }
    void load(java.io.InputStream in) throws java.io.IOException { props.load(in); }
    void store(java.io.OutputStream out, String comments) throws java.io.IOException { props.store(out, comments); }
}

// -----------------------------------------------------------------------------
// Epoch simulator - Simulate epoch completion for a run
// -----------------------------------------------------------------------------

final class EpochSimulator {
    private final int epochCount;
    private int currentEpoch;
    private final List<byte[]> gradientNorms = new ArrayList<>();

    EpochSimulator(int epochCount) {
        this.epochCount = epochCount;
        this.currentEpoch = 0;
    }

    boolean nextEpoch() {
        if (currentEpoch >= epochCount) return false;
        gradientNorms.add(HashUtils.sha256("grad-" + currentEpoch + "-" + System.nanoTime()));
        currentEpoch++;
        return true;
    }

    int getCurrentEpoch() { return currentEpoch; }
    List<byte[]> getGradientNorms() { return new ArrayList<>(gradientNorms); }
    boolean isComplete() { return currentEpoch >= epochCount; }
}

// -----------------------------------------------------------------------------
// Gradient snapshot cache - Cache gradient norm hashes per run/epoch
// -----------------------------------------------------------------------------

final class GradientSnapshotCache {
    private final Map<String, Map<Integer, byte[]>> cache = new ConcurrentHashMap<>();

    void put(String runId, int epochIndex, byte[] hash) {
        cache.computeIfAbsent(runId, k -> new ConcurrentHashMap<>()).put(epochIndex, hash);
    }

    byte[] get(String runId, int epochIndex) {
        Map<Integer, byte[]> runMap = cache.get(runId);
        return runMap == null ? null : runMap.get(epochIndex);
    }

    int countForRun(String runId) {
        Map<Integer, byte[]> runMap = cache.get(runId);
        return runMap == null ? 0 : runMap.size();
    }

    void clearRun(String runId) { cache.remove(runId); }
}

// -----------------------------------------------------------------------------
// Attestation batch - Batch multiple attestations for submission
// -----------------------------------------------------------------------------

final class AttestationBatch {
    private final List<String> runIds = new ArrayList<>();
    private final List<Boolean> approved = new ArrayList<>();

    void add(String runId, boolean approved) {
        runIds.add(runId);
        this.approved.add(approved);
    }

    int size() { return runIds.size(); }

    void submit(LocalYangGoRegistry registry, String validatorAddress) {
        for (int i = 0; i < runIds.size(); i++) {
            try {
                registry.attestRun(runIds.get(i), validatorAddress, approved.get(i));
            } catch (Exception ignored) { }
        }
    }
}

// -----------------------------------------------------------------------------
// Run query service - Query runs by tier, finalized, quorum
// -----------------------------------------------------------------------------

final class RunQueryService {
    private final LocalYangGoRegistry registry;

    RunQueryService(LocalYangGoRegistry registry) { this.registry = registry; }

    List<RunRecord> byModelTier(int tier) {
        return registry.getAllRuns().stream().filter(r -> r.getModelTier() == tier).collect(Collectors.toList());
    }

    List<RunRecord> finalizedOnly() {
        return registry.getAllRuns().stream().filter(RunRecord::isFinalized).collect(Collectors.toList());
    }

    List<RunRecord> withPositiveQuorum() {
        return registry.getAllRuns().stream()
            .filter(r -> QuorumCalculator.positiveQuorumReached(r.getPositiveAttestations(), r.getTotalAttestations()))
            .collect(Collectors.toList());
    }

    List<RunRecord> byCoordinator(String coordinator) {
        return registry.getAllRuns().stream().filter(r -> r.getCoordinator().equals(coordinator)).collect(Collectors.toList());
    }

    long countByTier(int tier) { return byModelTier(tier).size(); }
    long countFinalized() { return finalizedOnly().size(); }
}

// -----------------------------------------------------------------------------
// Validator weights - Optional weight per validator
// -----------------------------------------------------------------------------

final class ValidatorWeights {
    private final Map<String, Long> weights = new ConcurrentHashMap<>();

    void set(String address, long weight) { weights.put(address, weight); }
    long get(String address) { return weights.getOrDefault(address, 1L); }
    long getTotal(Collection<ValidatorState> validators) {
        return validators.stream().mapToLong(v -> get(v.getAddress())).sum();
    }
}

// -----------------------------------------------------------------------------
// Fee calculator - Compute registration fee by tier
// -----------------------------------------------------------------------------

final class FeeCalculator {
    private static final BigInteger BASE_WEI = BigInteger.valueOf(10_000_000_000_000_000L); // 0.01 ETH
    private static final int TIER_1_MUL = 10000, TIER_2_MUL = 12000, TIER_3_MUL = 15000, TIER_4_MUL = 20000;
    private static final int DENOM = 10000;

    static BigInteger feeForTier(int modelTier) {
        int mul = modelTier == 1 ? TIER_1_MUL : modelTier == 2 ? TIER_2_MUL : modelTier == 3 ? TIER_3_MUL : modelTier == 4 ? TIER_4_MUL : TIER_1_MUL;
        return BASE_WEI.multiply(BigInteger.valueOf(mul)).divide(BigInteger.valueOf(DENOM));
    }
}

// -----------------------------------------------------------------------------
// Preset manager - Config presets (label, tier, suggested epochs)
// -----------------------------------------------------------------------------

final class PresetRecord {
    private final String id;
    private final String label;
    private final int modelTier;
    private final int suggestedEpochs;

    PresetRecord(String id, String label, int modelTier, int suggestedEpochs) {
        this.id = id;
        this.label = label;
        this.modelTier = modelTier;
        this.suggestedEpochs = suggestedEpochs;
    }

    String getId() { return id; }
    String getLabel() { return label; }
    int getModelTier() { return modelTier; }
    int getSuggestedEpochs() { return suggestedEpochs; }
}

final class PresetManager {
    private final Map<String, PresetRecord> presets = new ConcurrentHashMap<>();

    void add(String id, String label, int modelTier, int suggestedEpochs) {
        presets.put(id, new PresetRecord(id, label, modelTier, suggestedEpochs));
    }

    PresetRecord get(String id) { return presets.get(id); }
    Collection<PresetRecord> getAll() { return new ArrayList<>(presets.values()); }
}

// -----------------------------------------------------------------------------
// Run metadata - Optional URI per run
// -----------------------------------------------------------------------------

final class RunMetadataStore {
    private final Map<String, String> uriByRunId = new ConcurrentHashMap<>();

    void set(String runId, String uri) { uriByRunId.put(runId, uri); }
    String get(String runId) { return uriByRunId.get(runId); }
}

// -----------------------------------------------------------------------------
// Tag index - Index runs by tag
// -----------------------------------------------------------------------------

final class TagIndex {
    private final Map<String, Set<String>> runIdsByTag = new ConcurrentHashMap<>();

    void index(String runId, String tag) {
        runIdsByTag.computeIfAbsent(tag, k -> ConcurrentHashMap.newKeySet()).add(runId);
    }

    Set<String> getRunIds(String tag) {
        Set<String> set = runIdsByTag.get(tag);
        return set == null ? Collections.emptySet() : new HashSet<>(set);
    }

    Set<String> getAllTags() { return new HashSet<>(runIdsByTag.keySet()); }
}

// -----------------------------------------------------------------------------
// Quorum stats aggregator
// -----------------------------------------------------------------------------

final class QuorumStatsAggregator {
    private final LocalYangGoRegistry registry;

    QuorumStatsAggregator(LocalYangGoRegistry registry) { this.registry = registry; }

    int totalRuns() { return registry.getRunCount(); }
    int totalValidators() { return registry.getValidatorCount(); }
    long runsWithQuorum() {
        return registry.getAllRuns().stream()
            .filter(r -> QuorumCalculator.quorumReached(r.getTotalAttestations(), registry.getValidatorCount()))
            .count();
    }
    long runsWithPositiveQuorum() {
        return registry.getAllRuns().stream()
            .filter(r -> QuorumCalculator.positiveQuorumReached(r.getPositiveAttestations(), r.getTotalAttestations()))
            .count();
    }
}

// -----------------------------------------------------------------------------
// Model tier labels
// -----------------------------------------------------------------------------

final class ModelTierLabels {
    static String name(int tier) {
        if (tier == 1) return "base";
        if (tier == 2) return "mid";
        if (tier == 3) return "large";
        if (tier == 4) return "xl";
        return "unknown";
    }

    static int index(String name) {
        if ("base".equalsIgnoreCase(name)) return 1;
        if ("mid".equalsIgnoreCase(name)) return 2;
        if ("large".equalsIgnoreCase(name)) return 3;
        if ("xl".equalsIgnoreCase(name)) return 4;
        return 0;
    }
}

// -----------------------------------------------------------------------------
// Run age calculator
// -----------------------------------------------------------------------------

final class RunAgeCalculator {
    static long ageSeconds(RunRecord r, long asOfMs) {
        long reg = r.getRegisteredAt();
        if (asOfMs <= reg) return 0;
        return (asOfMs - reg) / 1000;
    }

    static long ageSecondsNow(RunRecord r) {
        return ageSeconds(r, System.currentTimeMillis());
    }
}

// -----------------------------------------------------------------------------
// Checkpoint batch builder
// -----------------------------------------------------------------------------

final class CheckpointBatchBuilder {
    private final List<byte[]> hashes = new ArrayList<>();

    void add(byte[] hash) { hashes.add(hash); }
    void add(String label) { hashes.add(HashUtils.sha256(label)); }
    List<byte[]> build() { return new ArrayList<>(hashes); }
}

// -----------------------------------------------------------------------------
// Domain separator (EIP-712 style) - for off-chain signing
// -----------------------------------------------------------------------------

final class DomainSeparatorUtil {
    static String domainTypeHash() { return "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"; }
    static String attestationTypeHash() { return "Attest(uint256 runId,bool approved,uint256 nonce)"; }
}

// -----------------------------------------------------------------------------
// Nonce store per run (replay protection)
// -----------------------------------------------------------------------------

final class RunNonceStore {
    private final Map<String, AtomicLong> nonceByRunId = new ConcurrentHashMap<>();

    long get(String runId) {
        return nonceByRunId.computeIfAbsent(runId, k -> new AtomicLong(0)).get();
    }

    long increment(String runId) {
        return nonceByRunId.computeIfAbsent(runId, k -> new AtomicLong(0)).incrementAndGet();
    }
}

// -----------------------------------------------------------------------------
// Paginator for run list
// -----------------------------------------------------------------------------

final class RunPaginator {
    private final List<RunRecord> all;
    private final int pageSize;

    RunPaginator(Collection<RunRecord> all, int pageSize) {
        this.all = new ArrayList<>(all);
        this.pageSize = pageSize;
    }

    int pageCount() {
        if (pageSize <= 0) return 0;
        return (all.size() + pageSize - 1) / pageSize;
    }

    List<RunRecord> page(int pageIndex) {
        if (pageIndex < 0 || pageSize <= 0) return Collections.emptyList();
        int start = pageIndex * pageSize;
        if (start >= all.size()) return Collections.emptyList();
        int end = Math.min(start + pageSize, all.size());
        return new ArrayList<>(all.subList(start, end));
    }
}

// -----------------------------------------------------------------------------
// Epoch bucket classifier
// -----------------------------------------------------------------------------

final class EpochBucketUtil {
    static int getBucket(int epochCount) {
        if (epochCount <= 10) return 1;
        if (epochCount <= 100) return 2;
        if (epochCount <= 500) return 3;
        if (epochCount <= 1000) return 4;
        return 5;
    }

    static String getBucketLabel(int bucket) {
        if (bucket == 1) return "small";
        if (bucket == 2) return "medium";
        if (bucket == 3) return "large";
        if (bucket == 4) return "xlarge";
        if (bucket == 5) return "mega";
        return "unknown";
    }
}

// -----------------------------------------------------------------------------
// Run existence checker
// -----------------------------------------------------------------------------

final class RunExistenceChecker {
    private final LocalYangGoRegistry registry;

    RunExistenceChecker(LocalYangGoRegistry registry) { this.registry = registry; }

    boolean exists(String runId) { return registry.getRun(runId) != null; }
    int totalRuns() { return registry.getRunCount(); }
}

// -----------------------------------------------------------------------------
// Version info
// -----------------------------------------------------------------------------

final class YangGoVersionInfo {
    static final int VERSION = 2;
    static final String LABEL = "YangGo AI Training Registry v2";
}

// -----------------------------------------------------------------------------
// Batch run registration from CSV-like lines
// -----------------------------------------------------------------------------

final class CsvRunLoader {
    private final LocalYangGoRegistry registry;

    CsvRunLoader(LocalYangGoRegistry registry) { this.registry = registry; }

    int load(String line, String coordinator) {
        String[] parts = line.split(",");
        if (parts.length < 3) return 0;
        int tier = Integer.parseInt(parts[0].trim());
        int epochs = Integer.parseInt(parts[1].trim());
        String dsLabel = parts.length > 2 ? parts[2].trim() : "ds";
        byte[] dsHash = HashUtils.sha256(dsLabel);
        byte[] cfgHash = HashUtils.sha256("cfg-" + dsLabel);
        registry.registerRun(dsHash, cfgHash, tier, epochs, coordinator);
        return 1;
    }

    int loadAll(List<String> lines, String coordinator) {
        int count = 0;
        for (String line : lines) {
            if (line.trim().isEmpty() || line.trim().startsWith("#")) continue;
            count += load(line, coordinator);
        }
        return count;
    }
}

// -----------------------------------------------------------------------------
// Async run submitter (simulated delay)
// -----------------------------------------------------------------------------

final class AsyncRunSubmitter {
    private final LocalYangGoRegistry registry;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    AsyncRunSubmitter(LocalYangGoRegistry registry) { this.registry = registry; }

    Future<RunRecord> submitAsync(byte[] datasetHash, byte[] configHash, int modelTier, int epochCount, String coordinator) {
        return executor.submit(() -> {
            Thread.sleep(10);
            return registry.registerRun(datasetHash, configHash, modelTier, epochCount, coordinator);
        });
    }

    void shutdown() { executor.shutdown(); }
}

// -----------------------------------------------------------------------------
// Run summary DTO
// -----------------------------------------------------------------------------

final class RunSummaryDto {
    final String runId;
    final int modelTier;
    final int epochCount;
    final String coordinator;
    final boolean finalized;
    final int positiveAttestations;
    final int totalAttestations;
    final int checkpointCount;

    RunSummaryDto(RunRecord r) {
        this.runId = r.getRunId();
        this.modelTier = r.getModelTier();
        this.epochCount = r.getEpochCount();
        this.coordinator = r.getCoordinator();
        this.finalized = r.isFinalized();
        this.positiveAttestations = r.getPositiveAttestations();
        this.totalAttestations = r.getTotalAttestations();
        this.checkpointCount = r.getCheckpoints().size();
    }
}

// -----------------------------------------------------------------------------
// Validator summary DTO
// -----------------------------------------------------------------------------

final class ValidatorSummaryDto {
    final String address;
    final BigInteger stake;
    final int attestedCount;

    ValidatorSummaryDto(ValidatorState v) {
        this.address = v.getAddress();
        this.stake = v.getStake();
        this.attestedCount = v.getAttestedRuns().size();
    }
}

// -----------------------------------------------------------------------------
// Export runs to text report
// -----------------------------------------------------------------------------

final class RunReportExporter {
    static String exportAll(LocalYangGoRegistry registry) {
        StringBuilder sb = new StringBuilder();
        sb.append("YangGo Run Report\n");
        sb.append("Total runs: ").append(registry.getRunCount()).append("\n");
        sb.append("Total validators: ").append(registry.getValidatorCount()).append("\n\n");
        for (RunRecord r : registry.getAllRuns()) {
            sb.append("Run: ").append(r.getRunId()).append(" | tier=").append(r.getModelTier())
              .append(" | epochs=").append(r.getEpochCount()).append(" | final=").append(r.isFinalized())
              .append(" | attest=").append(r.getPositiveAttestations()).append("/").append(r.getTotalAttestations()).append("\n");
        }
        return sb.toString();
    }
}

// -----------------------------------------------------------------------------
// Scheduler stub for periodic attestation check
// -----------------------------------------------------------------------------

final class AttestationScheduler {
    private final LocalYangGoRegistry registry;
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

    AttestationScheduler(LocalYangGoRegistry registry) { this.registry = registry; }

    void scheduleAttestationCheck(Runnable check, long periodMs) {
        scheduler.scheduleAtFixedRate(check, periodMs, periodMs, TimeUnit.MILLISECONDS);
    }

    void shutdown() { scheduler.shutdown(); }
}

// -----------------------------------------------------------------------------
// Run ID generator (alternative)
// -----------------------------------------------------------------------------

final class RunIdGenerator {
    private static final AtomicLong counter = new AtomicLong(0);

    static String next() {
        return "run_" + counter.incrementAndGet() + "_" + System.currentTimeMillis() + "_" + Integer.toHexString(new Random().nextInt());
    }

    static String withPrefix(String prefix) {
        return prefix + "_" + counter.incrementAndGet() + "_" + System.currentTimeMillis();
    }
}

// -----------------------------------------------------------------------------
// Config hash registry (label per config hash)
// -----------------------------------------------------------------------------

final class ConfigHashRegistry {
    private final Map<String, String> labelByHashHex = new ConcurrentHashMap<>();

    void register(String configHashHex, String label) { labelByHashHex.put(configHashHex, label); }
    String getLabel(String configHashHex) { return labelByHashHex.get(configHashHex); }
}

// -----------------------------------------------------------------------------
// Dataset hash registry
// -----------------------------------------------------------------------------

final class DatasetHashRegistry {
    private final Map<String, String> labelByHashHex = new ConcurrentHashMap<>();

    void register(String datasetHashHex, String label) { labelByHashHex.put(datasetHashHex, label); }
    String getLabel(String datasetHashHex) { return labelByHashHex.get(datasetHashHex); }
}

// -----------------------------------------------------------------------------
// CmdExport - Export run report to stdout or file
// -----------------------------------------------------------------------------

final class CmdExport implements Command {
    @Override public String name() { return "export"; }
    @Override public String usage() { return "export [filename]"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        String report = RunReportExporter.exportAll(registry);
        if (args.isEmpty()) {
            out.println(report);
            return;
        }
        try (java.io.PrintWriter w = new java.io.PrintWriter(args.get(0), StandardCharsets.UTF_8)) {
            w.print(report);
        } catch (java.io.IOException e) {
            out.println("Error: " + e.getMessage());
        }
    }
}

// -----------------------------------------------------------------------------
// CmdQueryByTier - List runs by model tier
// -----------------------------------------------------------------------------

final class CmdQueryByTier implements Command {
    @Override public String name() { return "query-tier"; }
    @Override public String usage() { return "query-tier <1|2|3|4>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.isEmpty()) { out.println("Usage: " + usage()); return; }
        int tier = Integer.parseInt(args.get(0));
        RunQueryService q = new RunQueryService(registry);
        for (RunRecord r : q.byModelTier(tier)) {
            out.println(r.getRunId() + " | " + r.getEpochCount() + " epochs | final=" + r.isFinalized());
        }
        out.println("Count: " + q.byModelTier(tier).size());
    }
}

// -----------------------------------------------------------------------------
// CmdQueryQuorum - List runs with positive quorum
// -----------------------------------------------------------------------------

final class CmdQueryQuorum implements Command {
    @Override public String name() { return "query-quorum"; }
    @Override public String usage() { return "query-quorum"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        RunQueryService q = new RunQueryService(registry);
        for (RunRecord r : q.withPositiveQuorum()) {
            out.println(r.getRunId() + " | " + r.getPositiveAttestations() + "/" + r.getTotalAttestations());
        }
        out.println("Count: " + q.withPositiveQuorum().size());
    }
}

// -----------------------------------------------------------------------------
// CmdListValidators - List all validators
// -----------------------------------------------------------------------------

final class CmdListValidators implements Command {
    @Override public String name() { return "list-validators"; }
    @Override public String usage() { return "list-validators"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        for (ValidatorState v : registry.getAllValidators()) {
            out.println(v.getAddress() + " | stake=" + v.getStake() + " | attested=" + v.getAttestedRuns().size());
        }
        out.println("Total: " + registry.getValidatorCount());
    }
}

// -----------------------------------------------------------------------------
// CmdFee - Show fee for tier
// -----------------------------------------------------------------------------

final class CmdFee implements Command {
    @Override public String name() { return "fee"; }
    @Override public String usage() { return "fee <tier 1-4>"; }
    @Override
    public void run(List<String> args, LocalYangGoRegistry registry, Print out) {
        if (args.isEmpty()) { out.println("Usage: " + usage()); return; }
        int tier = Integer.parseInt(args.get(0));
        BigInteger fee = FeeCalculator.feeForTier(tier);
        out.println("Fee for tier " + tier + ": " + fee + " wei");
