/*
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;*/

import org.datavec.image.mnist.MnistDbFile;
import org.datavec.image.mnist.MnistImageFile;
import org.datavec.image.mnist.MnistLabelFile;
import org.datavec.image.mnist.MnistManager;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.spark.sql.SparkSession;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.collection.Seq;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.List;

import java.util.logging.XMLFormatter;

public class Main {
    private static String appName = "Dl4jApplication";
    public static enum Master {local,remote}
    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws MalformedURLException {

        System.setProperty("log4j.configuration", new File("resources", "log4j.properties").toString());


        //Select between local and remote master
        Master selectMaster = Master.local;

        int batchSizePerWorker=16;
        int numEpochs = 3;

        //Creating Spark Session
        SparkSession ss;
        if (selectMaster==Master.local)
            ss = SparkSession.builder().master("local").appName(appName).getOrCreate();
        else
            ss =  SparkSession.builder().appName(appName).getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(ss.sparkContext());

        try {

            //1. Read Dataset (MNIST)

            //new MnistDbFile("../data/", "");
            //new MnistImageFile(String name, String mode);
            //new MnistLabelFile(String name, String mode);

            System.setProperty("user.home","./data");

            //Features: 16x784x784 (batch, height, width)
            //Labels:   16x10      (batch, nDigits)
            DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
            DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
            List<DataSet> trainDataList = new ArrayList<>();
            List<DataSet> testDataList = new ArrayList<>();
            while (iterTrain.hasNext()) {
                trainDataList.add(iterTrain.next());
            }
            while (iterTest.hasNext()) {
                testDataList.add(iterTest.next());
            }


            JavaRDD < DataSet > trainData = sc.parallelize(trainDataList);
            JavaRDD<DataSet> testData = sc.parallelize(testDataList);

            //2. Configure Neural network

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)

                    .activation(Activation.LEAKYRELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(0.02))// To configure: .updater(Nesterovs.builder().momentum(0.9).build())
                    .l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
                    .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
                    .pretrain(false).backprop(true)
                    .build();

            //Configuration for Spark training: see https://deeplearning4j.org/distributed for explanation of these configuration options
            VoidConfiguration voidConfiguration = VoidConfiguration.builder()

                    /**
                     * This can be any port, but it should be open for IN/OUT comms on all Spark nodes
                     */
                    //.unicastPort(40123)

                    /**
                     * if you're running this example on Hadoop/YARN, please provide proper netmask for out-of-spark comms
                     */
                   // .networkMask("10.1.1.0/24")

                    /**
                     * However, if you're running this example on Spark standalone cluster, you can rely on Spark internal addressing via $SPARK_PUBLIC_DNS env variables announced on each node
                     */
                    .controllerAddress((selectMaster==Master.local) ? "127.0.0.1" : null)
                    .build();

            TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSizePerWorker)
                    // encoding threshold. Please check https://deeplearning4j.org/distributed for details
                    .updatesThreshold(1e-3)
                    .rddTrainingApproach(RDDTrainingApproach.Direct)
                    .batchSizePerWorker(batchSizePerWorker)

                    // this option will enforce exactly 4 workers for each Spark node
                    .workersPerNode(4)
                    .build();

            //Create the Spark network
            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

            //Execute training:
            for (int i = 0; i < numEpochs; i++) {
                sparkNet.fit(trainData);
                log.info("Completed Epoch {}", i);
            }

            //Perform evaluation (distributed)
            //        Evaluation evaluation = sparkNet.evaluate(testData);
            Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
            log.info("***** Evaluation *****");
            log.info(evaluation.stats());

            //Delete the temp training files, now that we are done with them
            tm.deleteTempFiles(sc);

            log.info("***** Example Complete *****");

        }
        catch (IOException ex) {
            ex.printStackTrace();
        }

    }
}

