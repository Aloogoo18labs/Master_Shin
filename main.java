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
