package training;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


/**
 * Created by andreapasini on 06/05/18.
 * Distributed Dataset structure.
 * Methods:
 *  Read dataset (from serialized INDArray images) and divide into batches.
 * Dataset shape:
 *  NCHW convention, [minibatch, channel, height, width]
 */
public class DistributedDataset {
    private JavaSparkContext sc;                        //Java spark context
    private int batchSize;                              //batch size
    private int numBatches;                             //number of batches
    private long numSamples;                            //number of samples
    private Map<String, INDArray> labels;               //dataset labels
    private Broadcast<Map<String, INDArray>> bLabels;   //dataset labels (broadcast var)
    private JavaRDD<DataSet>  dataRDD;                  //dataset batches RDD<dataset>

    /**
     * Constructor
     * @param sc JavaSparkContext
     * @param batchSize the batch size for training (num images)
     * @param inputLabels input file with labels list
     * @param inputPath input RDD<filename,INDArray>
     */
    public DistributedDataset(JavaSparkContext sc, int batchSize, String inputLabels, String inputPath) throws IOException {
        this.sc = sc;
        this.batchSize = batchSize;
        loadLabels(inputLabels);
        loadSerializedDataset(inputPath);
    }

    /**
     * Open dataInputStream from path.
     */
    private DataInputStream getFileStream(String path) throws IOException {
        //Get filesystem object
        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);

        //Return DataInputStream
        return fs.open(new Path(path));
    }

    /**
     * Load labels from file (list of labels)
     * Generate 1-hot vectors.
     */
    private Map<String, INDArray> loadLabels(String inputFile) throws IOException {
        labels=new HashMap<>();
        DataInputStream dis = getFileStream(inputFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(dis));

        String line;
        //Read the list of labels
        List<String> labelList = new LinkedList<>();
        while ((line=br.readLine())!=null)
            labelList.add(line);
        int i=0;
        //Generate 1-hot vectors
        for (String label : labelList) {
            float[] label1Hot = new float[labelList.size()];
            label1Hot[i] = 1;
            labels.put(label, Nd4j.create(label1Hot));
            i++;
        }

        //Broadcast variable
        bLabels = sc.broadcast(labels);
        if (bLabels == null)
            throw new IOException();
        return labels;
    }

    /**
     * Read vectorized input images (file with RDD<filename, INDArray>)
     * Generate the training set batches: RDD<DataSet>
     * @param inputPath: path with the input RDD
     */
    private JavaRDD<DataSet> loadSerializedDataset(String inputPath) {

        //Read serialized images (NDArray)
        JavaRDD<Tuple2<String, INDArray>> binaryImagesRDD = sc.objectFile(inputPath);
        //Obtain an index for each sample
        JavaPairRDD<Tuple2<String, INDArray>, Long> indexedImagesRDD = binaryImagesRDD.zipWithIndex();
        indexedImagesRDD.cache();
        //Compute the number of samples and batches
        numSamples = indexedImagesRDD.count();
        numBatches = (int)Math.ceil(1.0*numSamples/batchSize);
        final int numBatchesFinal = numBatches;
        final Broadcast<Map<String,INDArray>> blabelsFinal = bLabels;

        //Divide samples into batches RDD<batchId, images>
        JavaPairRDD<Integer, Iterable<Tuple2<String, INDArray>>> batchesRDD = indexedImagesRDD.mapToPair(p -> {
            Long index = p._2;    //Sample index
            return new Tuple2<>((int)(index % numBatchesFinal), p._1);
        }).groupByKey();

        //Generating datasets, 1 for each batch: RDD<batchId,Dataset>
        dataRDD = batchesRDD.map(entry -> {
            LinkedList<INDArray> dsImages = new LinkedList<>();//IndArray for each sample (image)
            LinkedList<INDArray> dsLabels = new LinkedList<>();//IndArray for each sample (label)

            //Iterates over samples in the batch
            entry._2().forEach(t -> {
                String label = t._1.split("_")[1];//label from filename
                INDArray lind = blabelsFinal.value().get(label);//vectorized label
                dsImages.addLast(t._2);
                dsLabels.addLast(lind);
            });

            //Generate DataSet
            int[] featureShape = dsImages.get(0).shape(); //image shape
            int[] labelShape = dsLabels.get(0).shape();   //vectorized label shape
            return new DataSet(
                    Nd4j.create(dsImages,                 //features
                            new int[]{dsImages.size(),    //Shape=[numImages,nChannels,height,width]
                                featureShape[0],
                                featureShape[1],
                                featureShape[2]}),
                    Nd4j.create(dsLabels,                 //labels
                            new int[]{dsLabels.size(),
                                    labelShape[1]}));

        });

        return dataRDD;
    }

    //Getters
    public int getBatchSize() { return batchSize; }
    public int getNumBatches() { return numBatches; }
    public long getNumSamples() { return numSamples; }
    public Map<String, INDArray> getLabels() { return labels; }
    public Broadcast<Map<String, INDArray>> getLabelsBroadcast() { return bLabels; }
    public JavaRDD<DataSet> getDatasetRDD() { return dataRDD; }
}
