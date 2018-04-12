package datasets;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class DatasetBuilder {
    private JavaSparkContext sc;                    //Spark context
    private FileSystem fs;                          //File system manager
    private DirectoryIterator di;                   //Directory iterator (dataset files)
    private Map<String, INDArray> classLabels;      //Class labels and 1-hot vectors for this dataset

    /**
     * Constructor.
     * @param sparkContext java spark context.
     * @param datasetPath dataset path with image files
     */
    public DatasetBuilder(JavaSparkContext sparkContext, String datasetPath) throws IOException {
        this.sc = sparkContext;
        this.di = new DirectoryIterator(sc, datasetPath);
        Configuration conf = sc.hadoopConfiguration();
        fs = org.apache.hadoop.fs.FileSystem.get(conf);
    }

    /**
     * Read file with class label list and produces the Map<classLabel, oneHotVector>
     */
    public Map<String, INDArray> readClassLabels(String path) throws IOException {
        classLabels=new HashMap<>();
        DataInputStream dis = getFileStream(path);
        BufferedReader br = new BufferedReader(new InputStreamReader(dis));

        String line;
        List<String> labelList = new LinkedList<>();
        while ((line=br.readLine())!=null)
            labelList.add(line);
        int i=0;
        //Generate 1-hot vectors
        for (String label : labelList) {
            float[] label1Hot = new float[labelList.size()];
            label1Hot[i] = 1;
            classLabels.put(label, Nd4j.create(label1Hot));
            i++;
        }
        return classLabels;
    }

    /**
     * Open dataInputStream from path.
     */
    public DataInputStream getFileStream(String path) throws IOException {
        //Get filesystem object
        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);

        //Return DataInputStream
        return fs.open(new Path(path));
    }

    /**
     * Generate next batch.
     * @param batchSize the batch size (number of images)
     * @return
     */
    public DataSet nextBatch(int batchSize) throws IOException {
        List<INDArray> dsImages = new LinkedList<>();
        List<INDArray> dsLabels = new LinkedList<>();

        int i=0;
        ImageLoader imageLoader = new ImageLoader();
        while (di.hasNext() && i<batchSize){
            Path path = di.nextFile();

            //Get image label
            String label = path.getName().split("_")[1].split("\\.")[0];
            INDArray labelVect = classLabels.get(label);

            //Image vectorization
            DataInputStream dis = fs.open(path);
            INDArray img = imageLoader.asMatrix(dis);
            dis.close();

            //Add image and label to list
            dsImages.add(img);
            dsLabels.add(labelVect);

            i++;
        }
        //Generate DataSet
        int[] featureShape = dsImages.get(0).shape();
        int[] labelShape = dsLabels.get(0).shape();
        DataSet imageDataset = new DataSet(Nd4j.create(dsImages,new int[]{dsImages.size(),featureShape[0],featureShape[1]}),
                Nd4j.create(dsLabels, new int[]{dsLabels.size(), labelShape[1]}));

        return imageDataset;
    }
}
