import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Andrea Pasini
 * Francesco Ventura
 * Run a sample neural network on MNIST dataset.
 */
public class Test1 {
    private static Logger log = LoggerFactory.getLogger(Test1.class);
    private int batchSizePerWorker = 16;
    private int numEpochs = 3;
    private JavaSparkContext sc;
    private boolean runLocal;

    /**
     * Constructor.
     */
    public Test1(JavaSparkContext sc, boolean runLocal) {
        this.sc = sc;
        this.runLocal = runLocal;
    }

    /**
     * Run a sample neural network on MNIST dataset.
     */
    public void run(){


        try {

            //1. Read Dataset (MNIST) into Spark RDD
            System.setProperty("user.home","./data");
            log.info("Reading MNIST...");
            JavaRDD<DataSet> trainData = readMnist(true);
            JavaRDD<DataSet> testData = readMnist(false);
            log.info("Done.");

            //2. Configure Neural network
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .activation(Activation.LEAKYRELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(0.02))
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
                    .controllerAddress((runLocal) ? "127.0.0.1" : null)
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

    /**
     * Read Local Mnist into JavaRDD
     * train=true for reading training set, otherwise test set
     */
    private JavaRDD<DataSet> readMnist(boolean train) throws IOException {
        //Features: 16x784x784 (batch, height, width)
        //Labels:   16x10      (batch, nDigits)
        DataSetIterator iter = new MnistDataSetIterator(batchSizePerWorker, train, 12345);
        List<DataSet> dataList = new ArrayList<>();
        while (iter.hasNext())
            dataList.add(iter.next());
        return sc.parallelize(dataList);
    }
}
