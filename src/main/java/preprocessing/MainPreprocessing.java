package preprocessing;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.sql.SparkSession;
import org.datavec.image.loader.ImageLoader;
import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Product2;
import utils.Utils;
import scala.Tuple2;

import java.io.DataInputStream;
import java.util.Map;

/**
 * Created by francescoventura on 12/04/18.
 */
public class MainPreprocessing {

    private static String appName = "Dl4jPreprocessing";


    public static void main(String[] args) {

        System.out.println("Starting job...");
        //Creating Spark Session
        SparkSession ss;
        boolean runLocal = false;
        String localConf = null;

        if (args.length < 1) {
            System.out.println("<input_path>");
        }

        if (System.getenv("CLUSTER_TEST_LOCAL") != null) {
            runLocal = true;
            localConf = System.getenv("CLUSTER_TEST_LOCAL");
        }

        if (runLocal)
            ss = SparkSession.builder().master(localConf).appName(appName)
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    .config("spark.kryoserializer.buffer.mb", "4")
                    .config("spark.files.maxPartitionBytes", "2400000")
                    .getOrCreate();
        else
            ss = SparkSession.builder().appName(appName)
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    .config("spark.kryoserializer.buffer.mb", "4")
                    //.config("spark.files.maxPartitionBytes","24")
                    .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());

        System.out.println("Running preprocessing");

        //Reading dataset
        JavaPairRDD<String, PortableDataStream> binaryRDD = sc.binaryFiles(args[0]);

        JavaPairRDD<String, INDArray> res = binaryRDD.mapToPair(kds -> {
            String imageId = Utils.getFileNameFromURI(kds._1);
            PortableDataStream ds = kds._2;

            DataInputStream dis = ds.open();
            ImageLoader il = new ImageLoader();
            INDArray img = il.asMatrix(dis);
            dis.close();
            return new Tuple2<>(imageId, img);
        });

        String outFolder = Utils.getOutFolderName(MainPreprocessing.class.getName().replace(".", ""));
        if (runLocal) {
            res.coalesce(1).saveAsTextFile(outFolder);
            res.coalesce(1).saveAsTextFile(outFolder + "_text");

        } else {
            res.saveAsObjectFile(outFolder);
            //res.saveAsTextFile(outFolder + "_text");
        }


        //Long l = fileNames.count();
        //System.out.println("ciao");
        //Image vectorization
        /*JavaPairRDD<String,String> res = binaryRDD.mapValues(ds -> {
//            DataInputStream dis = ds.open();
//            ImageLoader il = new ImageLoader();
//            INDArray img = il.asMatrix(dis);
//            dis.close();
//
//            return img.shapeInfoToString();
            return "pippo";
        });*/
        //res.saveAsTextFile("./outputSpark");

    }
}
