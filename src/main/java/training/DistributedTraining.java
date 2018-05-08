package training;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.impl.GLMRegressionModel;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class DistributedTraining {
    private JavaSparkContext sc;                //spark context
    private DistributedDataset trainingSet;     //training set data
    private SparkDl4jMultiLayer model;          //model being trained
    private TrainingMaster tm;                  //training master
    private boolean runLocal;                   //true if running locally

    /**
     * Constructor, given spark context and dataset.
     */
    public DistributedTraining(JavaSparkContext sc, DistributedDataset trainingSet, boolean runLocal) {
        this.sc = sc;
        this.trainingSet = trainingSet;
        this.runLocal = runLocal;
    }

    public void setModel(MultiLayerConfiguration conf) {


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

        int batchSizePerWorker = 50;
        int numEpochs=1;


        tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSizePerWorker)
                // encoding threshold. Please check https://deeplearning4j.org/distributed for details
                .updatesThreshold(1e-3)
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .batchSizePerWorker(batchSizePerWorker)

                // this option will enforce exactly 4 workers for each Spark node
                .workersPerNode(4)
                .build();

        //Create the Spark network
        model = new SparkDl4jMultiLayer(sc, conf, tm);
    }






    public void train(int numEpochs) {
        System.out.println("Training model...");

        //Execute training:
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingSet.getDatasetRDD());
            System.out.println("Completed Epoch " + i);
        }

        //Perform evaluation (distributed)
        //        Evaluation evaluation = sparkNet.evaluate(testData);
        Evaluation evaluation = model.doEvaluation(trainingSet.getDatasetRDD(), 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
        // log.info("***** Evaluation *****");
        // log.info(evaluation.stats());
        System.out.println(evaluation.stats());
        System.out.println(evaluation.confusionToString());

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);
    }
}
