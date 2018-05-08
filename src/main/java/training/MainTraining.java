package training;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

/**
 * Created by francescoventura on 17/04/18.
 */
public class MainTraining {

    private static String appName = "Dl4jApplication";
    private static Integer NumPartitions = 10;
    private static boolean runLocal = false;
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

        run(sc, args[0], args[1]);
        System.out.println("Done.");
        System.exit(0);
    }

    static MultiLayerConfiguration getModelArchitecture() {
        int nChannels = 3;
        int imgWidth = 32;
        int imgHeight = 32;
        int outputNum = 10;

        return new NeuralNetConfiguration.Builder()
                //.seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(imgHeight,imgWidth,nChannels)) //See note below
                .backprop(true).pretrain(false).build();

    }

    /**
     * Workflow
     */
    static void run(JavaSparkContext sc, String inputPath, String inputLabels) throws IOException {
        //Read training set
        System.out.println("Reading Training Set...");
        DistributedDataset trainingSet = new DistributedDataset(sc, 100, inputLabels, inputPath);

        //Printing
//        trainingSet.getDatasetRDD().take(10).forEach(r -> {
//            System.out.println(String.format("Batch id:%d - size: %d -> %s",r._1,r._2.getFeatures().length(),r._2.getFeatures().shapeInfoToString()));
//            System.out.println();
//        });

        //Preparing for training
        DistributedTraining dTraining = new DistributedTraining(sc, trainingSet, runLocal);
        dTraining.setModel(getModelArchitecture());
        dTraining.train(1);
    }
}
