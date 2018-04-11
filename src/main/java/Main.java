
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.hash.Hash;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.partial.PartialResult;


import org.apache.spark.sql.SparkSession;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import javax.xml.crypto.Data;
import java.io.*;
import java.net.URI;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

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
    public static boolean runLocal = true;



    public static void main(String[] args) throws IOException {

        //Creating Spark Session
        SparkSession ss;
        if (runLocal)
            ss = SparkSession.builder().master("local").appName(appName).getOrCreate();
        else
            ss =  SparkSession.builder().appName(appName).getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());
        System.out.println("testprint");

        //Reading Class Labels
        Map<String, INDArray> labels;
        if (runLocal)
            labels = DatasetBuilder.readClassLabels(sc,"./data/cifar/labels.txt");
        else
            labels = DatasetBuilder.readClassLabels(sc,"hdfs://BigDataHA/user/pasini/data/cifar/labels.txt");

        for (Map.Entry<String, INDArray> e : labels.entrySet()){
            System.out.println(e.getKey()+" "+e.getValue());
        }


        //Reading training set images
        DirectoryIterator di = new DirectoryIterator();
        if (runLocal)
            di.initIterator("./data/cifar/train/",sc);
        else
            di.initIterator("hdfs://BigDataHA/user/pasini/data/cifar/train/",sc);
        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);

        List<INDArray> dsImages = new LinkedList<>();
        List<INDArray> dsLabels = new LinkedList<>();

        List<DataSet> images=new LinkedList<>();
        int i=0;
        ImageLoader imageLoader = new ImageLoader();
        while (di.hdfsIterator.hasNext()){
            LocatedFileStatus file = di.hdfsIterator.next();
            Path path = file.getPath();

            //Get image label
            String label = path.getName().split("_")[1].split("\\.")[0];
            INDArray labelVect = labels.get(label);

            //Image vectorization
            DataInputStream dis = fs.open(path);
            INDArray img = imageLoader.asMatrix(dis);
            dis.close();

            //Add image and label to list
            dsImages.add(img);
            dsLabels.add(labelVect);

            if (i>5)break;
            i++;
        }

        //Generate DataSet
        int[] featureShape = dsImages.get(0).shape();
        int[] labelShape = dsLabels.get(0).shape();
        DataSet imageDataset = new DataSet(Nd4j.create(dsImages,new int[]{dsImages.size(),featureShape[0],featureShape[1]}),
                Nd4j.create(dsLabels, new int[]{dsLabels.size(), labelShape[1]}));


        System.out.println("Features: " + imageDataset.getFeatures().shape()[0]+" "+imageDataset.getFeatures().shape()[1]+" "+imageDataset.getFeatures().shape()[2]+" ");
        System.out.println("Labels: " + imageDataset.getLabels().shape()[0]+" "+imageDataset.getLabels().shape()[1]);






        //JavaRDD<INDArray> imgRdd = sc.parallelize(images);
        //long num = imgRdd.count();


        //CifarDataSetIterator









        //Reading dataset
       /* JavaPairRDD<String, PortableDataStream> binaryRDD;
        if (runLocal)
            binaryRDD = sc.binaryFiles("./data/cifar/train/*");
        else
            binaryRDD = sc.binaryFiles("hdfs://BigDataHA/user/pasini/data/cifar/train/*");
        JavaRDD<String> fileNames=binaryRDD.sample(false, 0.001).map(x -> x._1());//countApprox(10000);
        fileNames.saveAsTextFile("outputSpark");
*/


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












/*


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
*/

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