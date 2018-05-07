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
    private DistributedDataset trainingSet;     //training set data



    /**
     * Constructor
     */
    public DistributedTraining(DistributedDataset trainingSet, JavaSparkContext sc) {
        this.trainingSet = trainingSet;



       /* trainingSet.getDatasetRDD().map(pair->pair._2()).take(10).forEach(ds->{
            System.out.println(ds.getFeatures().shapeInfoToString());
            System.out.println(ds.getLabels().shapeInfoToString());
        });
*/
        JavaRDD<DataSet> tset = trainingSet.getDatasetRDD().map(pair->pair._2()).flatMap(ds->{
            List<DataSet> dsets = new LinkedList<DataSet>();
            Iterator<DataSet> it = ds.iterator();
            while(it.hasNext()){
                dsets.add(it.next());
            }


            return dsets.iterator();
        });/*.take(10).forEach(ds->{
            System.out.println(ds.getFeatures().shapeInfoToString());
            System.out.println(ds.getLabels().shapeInfoToString());
        });*/

int nChannels = 1;
int outputNum = 100;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
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
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(28,28,nChannels)) //See note below
                .backprop(true).pretrain(false).build();


        boolean runLocal=true;

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
                sparkNet.fit(tset);
                //log.info("Completed Epoch {}", i);
            }

            //Perform evaluation (distributed)
            //        Evaluation evaluation = sparkNet.evaluate(testData);
            Evaluation evaluation = sparkNet.doEvaluation(trainingSet.getDatasetRDD().map(pair->pair._2()), 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
           // log.info("***** Evaluation *****");
           // log.info(evaluation.stats());

            //Delete the temp training files, now that we are done with them
            tm.deleteTempFiles(sc);

           // log.info("***** Example Complete *****");

























    }
}
