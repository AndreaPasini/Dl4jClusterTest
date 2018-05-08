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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;
import scala.Product2;
import utils.Utils;
import scala.Tuple2;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * Created by francescoventura on 12/04/18.
 */
public class MainPreprocessing {

    private static String appName = "Dl4jPreprocessing";

    /**
     * This method takes as input a dataset of images in form of ".har" file.
     *   Since reading millions of files is very slow with HDFS, these are grouped together into a hadoop archive (.har).
     *   The archive can be created with: "hadoop archive" command.
     * Output: produces a RDD<"imageName",INDArray> (serialized images, ready for dl4j usage)
     *
     * Usage:
     * @param args: arg[0]=input path with files to be merged
     */
    public static void main(String[] args) {

        System.out.println("Starting job...");
        //Creating Spark Session
        SparkSession ss;
        boolean runLocal = false;
        String localConf = null;

        if (args.length < 1) {
            System.out.println("<input_path>");
        }

        //Get local configuration from spark environment variables
        if (System.getenv("CLUSTER_TEST_LOCAL") != null) {
            runLocal = true;
            localConf = System.getenv("CLUSTER_TEST_LOCAL");
        }

        //Create spark session
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

        //Reading dataset file ".har" (archive of images)
        //Returns a RDD with the files into the archive. RDD<"filename",filestream>
        JavaPairRDD<String, PortableDataStream> binaryRDD = sc.binaryFiles(args[0]);
        //Image vectorization
        JavaRDD<Tuple2<String, INDArray>> res = binaryRDD.map(kds -> {
            String imageId = Utils.getFileNameFromURI(kds._1);
            PortableDataStream ds = kds._2;
            //generate INDArray from image data
            DataInputStream dis = ds.open();
            ImageLoader il = new ImageLoader();
            INDArray img = il.toBgr(dis); //BGR tensor, shape [3,height,width]
            dis.close();

            return new Tuple2<>(imageId, img);
        });

        //Save the generated RDD
        String outFolder = Utils.getOutFolderName(MainPreprocessing.class.getName().replace(".", ""));
        if (runLocal) {
            //only 1 output partition
            res.coalesce(1).saveAsObjectFile(outFolder);
        } else {
            res.saveAsObjectFile(outFolder);
        }

    }
}
