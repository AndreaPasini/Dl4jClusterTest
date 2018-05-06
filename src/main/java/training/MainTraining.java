package training;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;

/**
 * Created by francescoventura on 17/04/18.
 */
public class MainTraining {

    private static String appName = "Dl4jApplication";
    private static Integer NumPartitions = 10;

    /**
     *
     * Input path: should contain files with RDD<"filename",INDArray> (vectorized images)
     * @param args: args[0]=input path
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        System.out.println("Starting job...");
        //Creating Spark Session
        SparkSession ss;
        boolean runLocal = false;
        String localConf = null;

        //Input path:
        if (args.length < 1)
            System.out.println("<input_path>");
        //Spark configuration
        if (System.getenv("CLUSTER_TEST_LOCAL") != null) {
            runLocal = true;
            localConf = System.getenv("CLUSTER_TEST_LOCAL");
        }
        //Create Spark Session
        if (runLocal)
            ss = SparkSession.builder().master(localConf).appName(appName)
                    //KYRO SERIALIZER SEEMS NOT WORKING
                    //.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    //.config("spark.kryoserializer.buffer.mb", "4")
                    .config("spark.files.maxPartitionBytes", "2400000")
                    .getOrCreate();
        else
            ss = SparkSession.builder().appName(appName)
                    //KYRO SERIALIZER SEEMS NOT WORKING
                    //.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    //.config("spark.kryoserializer.buffer.mb", "4")
                    //.config("spark.files.maxPartitionBytes","24")
                    .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());

        System.out.println("Running preprocessing");

        run(sc, args[0], args[1]);
        return;
    }

    /**
     * Workflow
     */
    static void run(JavaSparkContext sc, String inputPath, String inputLabels) throws IOException {
        //Create dataset reader
        DistributedDataset dDataset = new DistributedDataset(sc, 100);

        //Reading dataset
        dDataset.loadLabels(inputLabels);
        JavaPairRDD<Integer,DataSet>  datasetRDD = dDataset.loadSerializedDataset(inputPath);

        datasetRDD.take(10).forEach(r -> {
            System.out.println(String.format("Batch id:%d - size: %d -> %s",r._1,r._2.getFeatures().length(),r._2.getFeatures().shapeInfoToString()));
            System.out.println();
        });

    }
}
