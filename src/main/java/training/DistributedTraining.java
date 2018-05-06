package training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DistributedTraining {
    private DistributedDataset trainingSet;     //training set data



    /**
     * Constructor
     */
    public DistributedTraining(DistributedDataset trainingSet) {
        this.trainingSet = trainingSet;









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

        boolean runLocal=true;

        //Configuration for Spark training: see https://deeplearning4j.org/distributed for explanation of these configuration options
//        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
//
//        /**
//         * This can be any port, but it should be open for IN/OUT comms on all Spark nodes
//         */
//        //.unicastPort(40123)
//
//        /**
//         * if you're running this example on Hadoop/YARN, please provide proper netmask for out-of-spark comms
//         */
//        // .networkMask("10.1.1.0/24")
//
//        /**
//         * However, if you're running this example on Spark standalone cluster, you can rely on Spark internal addressing via $SPARK_PUBLIC_DNS env variables announced on each node
//         */
//                    .controllerAddress((runLocal) ? "127.0.0.1" : null)
//                    .build();
//
//            TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSizePerWorker)
//                    // encoding threshold. Please check https://deeplearning4j.org/distributed for details
//                    .updatesThreshold(1e-3)
//                    .rddTrainingApproach(RDDTrainingApproach.Direct)
//                    .batchSizePerWorker(batchSizePerWorker)
//
//                    // this option will enforce exactly 4 workers for each Spark node
//                    .workersPerNode(4)
//                    .build();
//
//            //Create the Spark network
//            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
//
//            //Execute training:
//            for (int i = 0; i < numEpochs; i++) {
//                sparkNet.fit(trainData);
//                log.info("Completed Epoch {}", i);
//            }
//
//            //Perform evaluation (distributed)
//            //        Evaluation evaluation = sparkNet.evaluate(testData);
//            Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
//            log.info("***** Evaluation *****");
//            log.info(evaluation.stats());
//
//            //Delete the temp training files, now that we are done with them
//            tm.deleteTempFiles(sc);
//
//            log.info("***** Example Complete *****");
//
//
//
//
//
//
//
//
//
















    }
}
