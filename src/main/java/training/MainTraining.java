package training;

import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.datavec.image.loader.ImageLoader;
import org.glassfish.jersey.server.Broadcaster;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Function1;
import scala.Function2;
import scala.Tuple2;
import scala.Tuple3;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static datasets.DatasetBuilder.readClassLabels;

/**
 * Created by francescoventura on 17/04/18.
 */
public class MainTraining {

    private static String appName = "Dl4jApplication";
    private static Integer NumPartitions = 10;

    public static void main(String[] args) throws IOException {
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
                    //NON USARE KRYO MAAAAAAI!!!!!!
                    //.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    //.config("spark.kryoserializer.buffer.mb", "4")
                    .config("spark.files.maxPartitionBytes", "2400000")
                    .getOrCreate();
        else
            ss = SparkSession.builder().appName(appName)
                    //.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    //.config("spark.kryoserializer.buffer.mb", "4")
                    //.config("spark.files.maxPartitionBytes","24")
                    .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());

        System.out.println("Running preprocessing");

        //Reading dataset

        JavaRDD<Tuple2<String, INDArray>> binaryRDD = sc.objectFile(args[0]);
        binaryRDD.take(10)
                .forEach(r -> {
                    System.out.println(r._1 + " - " + r._2.shapeInfoToString());
                });

        JavaPairRDD<Integer, Iterable<Tuple2<String, INDArray>>> indexedRDD = binaryRDD.mapToPair(r -> {
            Integer key = Integer.parseInt(r._1.split("_")[0]);
            return new Tuple2<>(key % NumPartitions, new Tuple2<>(r._1, r._2));
        }).groupByKey();


        indexedRDD.take(10).forEach(r -> {
            System.out.print(r._1 + " - ");
            r._2.forEach(r2->System.out.print(r2._2.shapeInfoToString()));
            System.out.println();
        });

        Map<String, INDArray> labels = readClassLabels("file:///Users/francescoventura/IdeaProjects/Dl4jClusterTest/data/cifar/labels.txt", sc);
        final Broadcast<Map<String, INDArray>> bLabels = sc.broadcast(labels);
        if (bLabels == null) {
            throw new IOException();
        }



        JavaPairRDD<Integer,DataSet> datsetRDD = indexedRDD.mapValues(x -> {
            LinkedList<INDArray> dsImages = new LinkedList<>();
            LinkedList<INDArray> dsLabels = new LinkedList<>();

            x.forEach(t -> {
                String label = t._1.split("_")[1];
                INDArray lind = bLabels.value().get(label);
                dsImages.addLast(t._2);
                dsLabels.addLast(lind);

            });

            //Generate DataSet
            int[] featureShape = dsImages.get(0).shape();
            int[] labelShape = dsLabels.get(0).shape();
            DataSet imageDataset = new DataSet(Nd4j.create(dsImages,
                    new int[]{dsImages.size(),
                            featureShape[0],
                            featureShape[1]}),
                    Nd4j.create(dsLabels,
                            new int[]{dsLabels.size(),
                                    labelShape[1]}));

            return imageDataset;
        });

        datsetRDD.foreach(r -> {
            System.out.println(String.format("Batch id:%d - size: %d -> %s",r._1,r._2.getFeatures().length(),r._2.getFeatures().shapeInfoToString()));
            System.out.println();
        });


        return;
    }
}
