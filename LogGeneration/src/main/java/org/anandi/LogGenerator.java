package org.anandi;

import org.anandi.snip.eventlog.EventLog;
import org.anandi.snip.eventlog.EventLogUtils;
import org.anandi.snip.eventlog.Trace;
import org.anandi.snip.snip.NoiseType;
import org.anandi.snip.snip.absence.AbsenceNoiseInjector;
import org.anandi.snip.snip.insertion.InsertionNoiseInjector;
import org.anandi.snip.snip.ordering.OrderingNoiseInjector;
import org.anandi.snip.snip.ordering.ShiftingNoiseInjector;
import org.anandi.snip.snip.ordering.SwappingNoiseInjector;
import org.anandi.snip.snip.substitution.SubstitutionNoiseInjector;

import java.util.*;

public class LogGenerator {

    private final List<Integer> LOG_SIZES;
    private String logInputPath;
    private String logOutputPath;
    private Set<String> activities = null;

    private static final List<NoiseType> NOISE_TYPES = Arrays.asList(NoiseType.ABSENCE, NoiseType.INSERTION,
            NoiseType.SUBSTITUTION, NoiseType.ORDERING); // absence, insertion, substitution, ordering, mixed
    private static final List<Double> NOISE_LEVELS = Arrays.asList(0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0); // 7 noise levels

    public LogGenerator(List<Integer> logSizes, String logInputPath, String logOutputPath) {
        this.LOG_SIZES = logSizes;
        this.logInputPath = logInputPath;
        this.logOutputPath = logOutputPath;
    }

    public void generateLogs() {
        Map<Integer, EventLog> cleanLogs = generateCleanLogs();
        List<String> noiseTypes = new ArrayList<>();
        for (NoiseType noiseType : NOISE_TYPES) {
            noiseTypes.add(noiseType.toString());
        }
        noiseTypes.add("MIXED");

        for (int logSize : LOG_SIZES) {
            String logPath = logOutputPath + "_" + logSize + ".xes";
            new EventLogUtils().generateXES(cleanLogs.get(logSize), logPath);
        }

        for (int i = 1; i <= 5; i++) { // 5 iterations
            Map<String, Map<Integer, Map<Double, EventLog>>> noisyLogs = generateNoisyLogs(cleanLogs);

            for (String noiseType : noiseTypes) {
                for (int logSize : LOG_SIZES) {
                    for (double noiseLevel : NOISE_LEVELS) {
                        String logPath = logOutputPath + "_" + logSize + "_" + noiseType + "_" + noiseLevel + "_" + i + ".xes";
                        new EventLogUtils().generateXES(noisyLogs.get(noiseType).get(logSize).get(noiseLevel), logPath);
                    }
                }
            }
        }


    }

    // generate clean logs
    public Map<Integer, EventLog> generateCleanLogs() {
        EventLog eventLog = new EventLogUtils().readXES(logInputPath);
        this.activities = eventLog.getActivities();

        Random random = new Random();
        List<Trace> cumulative = new ArrayList<>();            // grows: 0 -> 1000 -> 2000 -> 4000 ...
        Map<Integer, EventLog> logsBySize = new LinkedHashMap<>();    // snapshots per size

        int prev = 0;
        for (int target : LOG_SIZES) {
            int toAdd = target - prev; // 1000, then 1000, then 2000
            for (int i = 0; i < toAdd; i++) {
                int idx = random.nextInt(eventLog.size());
                // Deep-copy the sampled trace so cumulative is independent of the source
                cumulative.add(new Trace(new ArrayList<>(eventLog.get(idx))));
            }
            // Deep-copy snapshot so this EventLog is independent of future growth AND prior logs
            List<Trace> snapshot = deepCopyTraces(cumulative);
            logsBySize.put(target, new EventLog(snapshot));

            prev = target;
        }
        return logsBySize;
    }

    public Map<String, Map<Integer, Map<Double, EventLog>>> generateNoisyLogs(Map<Integer, EventLog> cleanLogsBySize) {
        Map<String, Map<Integer, Map<Double, EventLog>>> result = new HashMap<>();
        for (NoiseType noiseType : NOISE_TYPES) {
            result.put(noiseType.toString(), generateNoisyLogsPerNoiseType(cleanLogsBySize, noiseType));
        }
        result.put("MIXED", generateNoisyLogsPerNoiseType(cleanLogsBySize, null));
        return result;
    }

    public Map<Integer, Map<Double, EventLog>> generateNoisyLogsPerNoiseType(Map<Integer, EventLog> cleanLogsBySize, NoiseType noiseType) {
        Map<Integer, Trace> prevChangedTraces = new LinkedHashMap<>(); // index -> noisy trace
        Random random = new Random();


        Map<Integer, Map<Double, EventLog>> result = new LinkedHashMap<>();

        for (int size : cleanLogsBySize.keySet()) {
            Map<Double, EventLog> noisyLogsAtThisLevel = new LinkedHashMap<>();
            EventLog cleanLog = cleanLogsBySize.get(size);
            List<Trace> noisyTraces = deepCopyTraces(cleanLog);

            for (double noise : NOISE_LEVELS) {
//                System.out.println(prevChangedTraces.size());
                int noisyTarget = (int) Math.round(size * (noise / 100.0));
                int applied = 0;


                // apply previously changed noisy traces
                for (Map.Entry<Integer, Trace> entry : prevChangedTraces.entrySet()) {
                    if (applied < noisyTarget) {
                        int idx = entry.getKey();
                        if (idx < noisyTraces.size()) {
                            noisyTraces.set(idx, deepCopyTrace(entry.getValue()));
                            applied++;
                        }
                    } else {
                        break;
                    }
                }

                while (applied < noisyTarget) {
                    int idx = random.nextInt(noisyTraces.size());
                    if (!prevChangedTraces.containsKey(idx)) {
                        Trace noisyTrace = new Trace(applyNoise(noisyTraces.get(idx), noiseType));
                        noisyTraces.set(idx, noisyTrace);
                        prevChangedTraces.put(idx, noisyTrace);
                        applied++;
                    }
                }
                noisyLogsAtThisLevel.put(noise, new EventLog(deepCopyTraces(noisyTraces)));
            }
            result.put(size, noisyLogsAtThisLevel);
        }
        return result;
    }


    private List<Trace> deepCopyTraces(List<Trace> traces) {
        List<Trace> copy = new ArrayList<>(traces.size());
        for (List<String> t : traces) {
            copy.add(new Trace(t)); // copy inner list
        }
        return copy;
    }

    private Trace deepCopyTrace(List<String> trace) {
        return new Trace(new ArrayList<>(trace));
    }

    private List<String> applyNoise(List<String> trace, NoiseType noiseType) {
        if (noiseType == null) {
            Random random = new Random();
            NoiseType newNoiseType = NOISE_TYPES.get(random.nextInt(NOISE_TYPES.size()));
            return applyNoise(trace, newNoiseType);
        } else {
            if (noiseType.equals(NoiseType.ABSENCE)) {
                return applyAbsenceNoise(trace);
            } else if (noiseType.equals(NoiseType.INSERTION)) {
                return applyInsertionNoise(trace);
            } else if (noiseType.equals(NoiseType.ORDERING)) {
                return applyOrderingNoise(trace);
            } else if (noiseType.equals(NoiseType.SUBSTITUTION)) {
                return applySubstitutionNoise(trace);
            } else {
                Random random = new Random();
                NoiseType newNoiseType = NOISE_TYPES.get(random.nextInt(NOISE_TYPES.size()));
                return applyNoise(trace, newNoiseType);
            }
        }


    }

    private List<String> applyAbsenceNoise(List<String> trace) {
        AbsenceNoiseInjector absenceNoiseInjector = new AbsenceNoiseInjector();
        Trace modifyingTrace = new Trace(trace);
        absenceNoiseInjector.injectNoise(modifyingTrace, 1);
        return new ArrayList<>(modifyingTrace);
    }

    private List<String> applyInsertionNoise(List<String> trace) {
        InsertionNoiseInjector insertionNoiseInjector = new InsertionNoiseInjector(activities);
        Trace modifyingTrace = new Trace(trace);
        insertionNoiseInjector.injectNoise(modifyingTrace, 1);
        return new ArrayList<>(modifyingTrace);
    }

    private List<String> applyOrderingNoise(List<String> trace) {
        Random random = new Random();
        double swappingProbability = 0.5;
        OrderingNoiseInjector orderingNoiseInjector;
        if (random.nextDouble() < swappingProbability) {
            orderingNoiseInjector = new SwappingNoiseInjector();
        } else {
            orderingNoiseInjector = new ShiftingNoiseInjector();
        }
        Trace modifyingTrace = new Trace(trace);
        orderingNoiseInjector.injectNoise(modifyingTrace, 1);
        return new ArrayList<>(modifyingTrace);
    }

    private List<String> applySubstitutionNoise(List<String> trace) {
        SubstitutionNoiseInjector substitutionNoiseInjector = new SubstitutionNoiseInjector(activities);
        Trace modifyingTrace = new Trace(trace);
        substitutionNoiseInjector.injectNoise(modifyingTrace, 1);
        return new ArrayList<>(modifyingTrace);
    }





}
