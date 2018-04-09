
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;

/**
 * Andrea Pasini
 * Francesco Ventura
 * Spark + Dl4j
 * Main class with spark session handling.
 */
public class Main {
    private static String appName = "Dl4jApplication";
    //private static Logger log = LoggerFactory.getLogger(Main.class);

    //Select between local and remote master
    public static boolean runLocal = false;

    public static void main(String[] args) throws IOException {

        //Creating Spark Session
        SparkSession ss;
        if (runLocal)
            ss = SparkSession.builder().master("local").appName(appName).getOrCreate();
        else
            ss =  SparkSession.builder().appName(appName).getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());
        System.out.println("testprint");

        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);
        boolean exists = fs.exists(new org.apache.hadoop.fs.Path("hdfs://BigDataHA/user/pasini/data/MNIST/t10k-images-idx3-ubyte"));
        if (exists) {
            System.out.println("ce");//ok lo trova
            //log.info("celog");
        } else {
            System.out.println("nonce");
            // log.info("non celog");
        }

        DataInputStream dis = fs.open(new org.apache.hadoop.fs.Path("hdfs://BigDataHA/user/pasini/data/MNIST/t10k-images-idx3-ubyte"));


        ////Read binary of mnist...
        




        /*
        //log.info("Spark Session created.");
        System.out.println("orava");*/
        /*try {
            File test = new File("hdfs://user/pasini/data/MNIST/t10k-images-idx3-ubyte");
            if (test.exists()) {
                System.out.println("ce");
                //log.info("celog");
            } else {
                System.out.println("nonce");
               // log.info("non celog");
            }
        }
        catch (Exception ex){
            System.out.println("eccezione");
            //log.info("non celog");
        }*/

        //Executing test:
        //Test1 test1 = new Test1(sc, runLocal);
        //test1.run();

        //Closing Spark Session
        ss.close();
        //log.info("Spark Session closed.");
      //  System.exit(0);
    }
}