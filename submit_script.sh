# Loading jar to server
scp target/ClusterTest1-1.0-jar-with-dependencies.jar ventura@bigdatalab.polito.it:dl4j
scp target/ClusterTest1-1.0-jar-with-dependencies.jar pasini@bigdatalab.polito.it:deeplearningTest

# HAR
hadoop archive -archiveName cifar_train.har -p /user/pasini/data/cifar/train /user/ventura/dataset/cifar

# Copying archive to local folder
hadoop fs -copyToLocal /user/ventura/dataset/cifar/cifar_train.har .
scp -r pasini@bigdatalab.polito.it:deeplearningTest/cifar_train.har cifar/

# Preprocessing
spark2-submit \
--master yarn \
--deploy-mode cluster \
--conf "spark.files.maxPartitionBytes=24000000" \
--class preprocessing.MainPreprocessing \
dl4j/ClusterTest1-1.0-jar-with-dependencies.jar har:///user/ventura/dataset/cifar/cifar_train.har

# Training
spark2-submit \
--master yarn \
--deploy-mode cluster \
--conf "spark.files.maxPartitionBytes=24000000" \
--class preprocessing.MainTraining \
dl4j/ClusterTest1-1.0-jar-with-dependencies.jar ./data/serializedTraining file:///Users/francescoventura/IdeaProjects/Dl4jClusterTest/data/cifar/labels.txt
