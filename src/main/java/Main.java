
import datasets.DatasetBuilder;
import datasets.DirectoryIterator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;


import org.apache.spark.sql.SparkSession;
import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import java.io.*;
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

        DatasetBuilder dsb;
        if (runLocal)
            dsb = new DatasetBuilder(sc,"./data/cifar/train/");
        else
            dsb = new DatasetBuilder(sc,"hdfs://BigDataHA/user/pasini/data/cifar/train/");



        //Reading Class Labels
        Map<String, INDArray> labels;
        if (runLocal)
            labels = dsb.readClassLabels("./data/cifar/labels.txt");
        else
            labels = dsb.readClassLabels("hdfs://BigDataHA/user/pasini/data/cifar/labels.txt");

        for (Map.Entry<String, INDArray> e : labels.entrySet()){
            System.out.println(e.getKey()+" "+e.getValue());
        }


        //Reading training set images
        List<DataSet> batches = new LinkedList<>();
        for (int i=0; i<50; i++) {
            batches.add(dsb.nextBatch(1000));
            System.out.println("batch: "+i+" "+batches.get(batches.size()-1).getFeatures().shape()[0]);
        }

        System.out.println("start parallelization");

        //Create RDD with the dataset
        JavaRDD<DataSet> datasetRDD = sc.parallelize(batches);
        datasetRDD.map(d-> d.getFeatures().shape()[0]+" "+d.getFeatures().shape()[1]+" "+d.getFeatures().shape()[2]).saveAsTextFile("cifarOutput");

        System.out.println("end parallelization");





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