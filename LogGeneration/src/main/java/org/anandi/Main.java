package org.anandi;

import org.anandi.snip.eventlog.EventLog;
import org.anandi.snip.eventlog.EventLogUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    // 1. dataset creation
    /*
    what to do?
    1. take 3 public datasets (BPI challenge 2012, RTFMS, Sepsis)
    2. generate 4 log samples (sampling with replacements) of sizes 1000, 2000, 4000, 10000 (3x4 = 12)
        generate 1000, then take that and add 1000, ...
    3. now inject noise (5 noise levels, 7 noise levels 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0) (35x12=)
        generate noisy versions (better if noise levels are part of the previous levels as well)
     */

    private static final List<String> SYSTEMS = Arrays.asList("Sepsis", "RTFMS", "BPIC2012");
    private static String systemInputPath = "src/main/resources/systems/";
    private static String outputPath = "src/main/resources/logs/";
    private static final List<Integer> LOG_SIZES = Arrays.asList(1000, 2000, 4000, 10000, 20000, 40000, 100000); // 7 log sizes
//    private static final List<Integer> LOG_SIZES = Arrays.asList(1000, 2000, 3000, 4000, 10000, 20000, 40000, 100000);

    public static void main(String[] args) {

        for (String system : SYSTEMS) {
            System.out.println(new EventLogUtils().readXES(systemInputPath + system + ".xes").size());
            new LogGenerator(LOG_SIZES, systemInputPath + system + ".xes", outputPath + system).generateLogs();
        }
    }


}