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
