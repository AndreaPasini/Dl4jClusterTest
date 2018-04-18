#
scp target/ClusterTest1-1.0-jar-with-dependencies.jar ventura@bigdatalab.polito.it:dl4j

#HAR
hadoop archive -archiveName cifar_train.har -p /user/pasini/data/cifar/train /user/ventura/dataset/cifar

# Preprocessing

spark2-submit \
--master yarn \
--deploy-mode cluster \
--conf "spark.files.maxPartitionBytes=24000000" \
--class preprocessing.MainPreprocessing \
dl4j/ClusterTest1-1.0-jar-with-dependencies.jar har:///user/ventura/dataset/cifar/cifar_train.har
