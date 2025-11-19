package org.anandi;

import org.anandi.snip.eventlog.EventLog;
import org.anandi.snip.eventlog.Trace;
import org.anandi.snip.snip.NoiseType;
import org.junit.jupiter.api.Test;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class LogGeneratorTest {

    @Test
    void testGenerateCleanLogs() {
        List<Integer> LOG_SIZES = Arrays.asList(1, 2, 4);
        // Arrange
        LogGenerator generator = new LogGenerator(LOG_SIZES,
                "src/main/resources/systems/Sepsis.xes", "src/main/resources/logs");

        // Act
        Map<Integer, EventLog> logsBySize = generator.generateCleanLogs();

        // Assert 1: sizes are correct
        for (int size : LOG_SIZES) {
            EventLog log = logsBySize.get(size);
            assertEquals(size, log.size(), "Log size mismatch for " + size);
        }

        // Assert 2: cumulative
        List<Trace> prevTraces = new ArrayList<>();
        for (int size : LOG_SIZES) {
            EventLog log = logsBySize.get(size);
            // Check that previous traces are contained at start
            for (int i = 0; i < prevTraces.size(); i++) {
                assertEquals(prevTraces.get(i), log.get(i), "Trace mismatch at cumulative index " + i);
            }
            prevTraces = log; // update for next iteration
        }

        // Assert 3: deep copy (modifying one log does not affect others)
        EventLog log1 = logsBySize.get(1);
        EventLog log2 = logsBySize.get(2);

        // Modify a trace in log1
        Trace firstTrace = log1.get(0);
        firstTrace.set(0, "MODIFIED");

        // Check that log2's first trace did NOT change
        assertNotEquals("MODIFIED", log2.get(0).get(0), "Deep copy failed: log2 affected by log1");
    }

    @Test
    void testNoisyLogs() {
        List<Integer> LOG_SIZES = Arrays.asList(1000, 2000, 4000);
        List<Double> NOISE_LEVELS = Arrays.asList(0.1, 0.2, 0.4, 1.0);
        List<String> NOISE_TYPES = Arrays.asList("ABSENCE", "INSERTION",
                "SUBSTITUTION", "ORDERING", "MIXED");

        LogGenerator generator = new LogGenerator(LOG_SIZES,
                "src/main/resources/systems/Sepsis.xes", "src/main/resources/logs");
        Map<Integer, EventLog> cleanLogs = generator.generateCleanLogs();
        Map<String, Map<Integer, Map<Double, EventLog>>> noisyLogs = generator.generateNoisyLogs(cleanLogs);

        for (String noiseType : NOISE_TYPES) {
            Map<Integer, Map<Double, EventLog>> logsByNoiseType = noisyLogs.get(noiseType);

            for (int size : LOG_SIZES) {
                Map<Double, EventLog> logsByNoise = logsByNoiseType.get(size);
                for (double noise : NOISE_LEVELS) {
                    EventLog noisyLog = logsByNoise.get(noise);
                    EventLog cleanLog = cleanLogs.get(size);

                    int expectedNoisy = (int) Math.round(size * (noise / 100.0));
                    int diffCount = 0;
                    for (int i = 0; i < cleanLog.size(); i++) {
                        if (!cleanLog.get(i).equals(noisyLog.get(i))) diffCount++;
                    }
                    assertEquals(expectedNoisy, diffCount,
                            "Noise level " + noise + "% does not match changed traces for log size " + size);
                }
            }
        }

        for (String noiseType : NOISE_TYPES) {
            Map<Integer, Map<Double, EventLog>> logsByNoiseType = noisyLogs.get(noiseType);

            for (int size : LOG_SIZES) {
                Map<Double, EventLog> logsByNoise = logsByNoiseType.get(size);
                List<Double> noiseLevels = new ArrayList<>(NOISE_LEVELS);
                for (int i = 1; i < noiseLevels.size(); i++) {
                    double prevNoise = noiseLevels.get(i - 1);
                    double currNoise = noiseLevels.get(i);

                    EventLog prevLog = logsByNoise.get(prevNoise);
                    EventLog currLog = logsByNoise.get(currNoise);

                    for (int j = 0; j < prevLog.size(); j++) {
                        Trace prevTrace = prevLog.get(j);
                        Trace currTrace = currLog.get(j);

                        // If trace was changed in previous noise level, it must remain changed
                        if (!cleanLogs.get(size).get(j).equals(prevTrace)) {
                            assertEquals(prevTrace, currTrace,
                                    "Trace " + j + " from previous noise level " + prevNoise +
                                            "% not carried forward to level " + currNoise + "% for log size " + size);
                        }
                    }
                }
            }
        }
    }

    @Test
    void testNoisyLogsStepByStep() {
        // Act
        List<Integer> LOG_SIZES = Arrays.asList(1000, 2000, 4000);
        List<Double> NOISE_LEVELS = Arrays.asList(0.1, 0.2, 0.4, 1.0);
        LogGenerator generator = new LogGenerator(LOG_SIZES,
                "src/main/resources/systems/Sepsis.xes", "src/main/resources/logs");
        Map<Integer, EventLog> cleanLogs = generator.generateCleanLogs();
        Map<Integer, Map<Double, EventLog>> noisyLogs = generator.generateNoisyLogsPerNoiseType(cleanLogs, NoiseType.ABSENCE);

        // Assert 1: clean logs unchanged
//        int traceCounter = 0;
//        for (int size : cleanLogs.keySet()) {
//            EventLog log = cleanLogs.get(size);
//            for (Trace t : log) {
//                assertEquals("Trace" + traceCounter, t.get(0), "Clean log modified!");
//                traceCounter++;
//            }
//        }

        // Assert 2: each noise level produces expected number of modified traces
//        for (double noise : NOISE_LEVELS) {
//            Map<Integer, EventLog> logsBySize = noisyLogs.get(noise);
//            for (int size : logsBySize.keySet()) {
//                EventLog log = logsBySize.get(size);
//                int expectedNoisyCount = (int) Math.round(size * noise / 100.0);
//                int actualNoisy = 0;
//                for (Trace t : log) {
//                    if (t.get(0).startsWith("NOISE_")) actualNoisy++;
//                }
//                assertEquals(expectedNoisyCount, actualNoisy,
//                        "Incorrect number of noisy traces at size " + size + " noise " + noise);
//            }
//        }

//        // Assert 3: modifying one noisy log does not affect another
//        EventLog smallLog = noisyLogs.get(NOISE_LEVELS.get(0)).get(1);
//        Trace t0 = smallLog.get(0);
//        t0.set(0, "MODIFIED");
//        EventLog nextLog = noisyLogs.get(NOISE_LEVELS.get(0)).get(2);
//        assertNotEquals("MODIFIED", nextLog.get(0).get(0), "Logs not independent");
//
//        // Assert 4: larger logs correctly include smaller logs' noisy traces
//        EventLog log2 = noisyLogs.get(NOISE_LEVELS.get(0)).get(2);
//        EventLog log1 = noisyLogs.get(NOISE_LEVELS.get(0)).get(1);
//        for (int i = 0; i < log1.size(); i++) {
//            assertEquals(log1.get(i), log2.get(i), "Cumulative noisy traces not carried forward");
//        }

        for (int size : LOG_SIZES) {
            Map<Double, EventLog> logsByNoise = noisyLogs.get(size);
            for (double noise : NOISE_LEVELS) {
                EventLog noisyLog = logsByNoise.get(noise);
                EventLog cleanLog = cleanLogs.get(size);

                int expectedNoisy = (int) Math.round(size * (noise / 100.0));
                int diffCount = 0;
                for (int i = 0; i < cleanLog.size(); i++) {
                    if (!cleanLog.get(i).equals(noisyLog.get(i))) diffCount++;
                }
                assertEquals(expectedNoisy, diffCount,
                        "Noise level " + noise + "% does not match changed traces for log size " + size);
            }
        }

        for (int size : LOG_SIZES) {
            Map<Double, EventLog> logsByNoise = noisyLogs.get(size);
            List<Double> noiseLevels = new ArrayList<>(NOISE_LEVELS);
            for (int i = 1; i < noiseLevels.size(); i++) {
                double prevNoise = noiseLevels.get(i - 1);
                double currNoise = noiseLevels.get(i);

                EventLog prevLog = logsByNoise.get(prevNoise);
                EventLog currLog = logsByNoise.get(currNoise);

                for (int j = 0; j < prevLog.size(); j++) {
                    Trace prevTrace = prevLog.get(j);
                    Trace currTrace = currLog.get(j);

                    // If trace was changed in previous noise level, it must remain changed
                    if (!cleanLogs.get(size).get(j).equals(prevTrace)) {
                        assertEquals(prevTrace, currTrace,
                                "Trace " + j + " from previous noise level " + prevNoise +
                                        "% not carried forward to level " + currNoise + "% for log size " + size);
                    }
                }
            }
        }

    }

}

