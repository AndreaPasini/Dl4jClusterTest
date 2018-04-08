
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Andrea Pasini
 * Francesco Ventura
 * Spark + Dl4j
 * Main class with spark session handling.
 */
public class Main {
    private static String appName = "Dl4jApplication";
    private static Logger log = LoggerFactory.getLogger(Main.class);

    //Select between local and remote master
    public static boolean runLocal = true;

    public static void main(String[] args) {

        //Creating Spark Session
        SparkSession ss;
        if (runLocal)
            ss = SparkSession.builder().master("local").appName(appName).getOrCreate();
        else
            ss =  SparkSession.builder().appName(appName).getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());
        log.info("Spark Session created.");

        //Executing test:
        Test1 test1 = new Test1(sc, runLocal);
        test1.run();

        //Closing Spark Session
        ss.close();
        log.info("Spark Session closed.");
        System.exit(0);
    }
}