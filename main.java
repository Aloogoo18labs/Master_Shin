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
