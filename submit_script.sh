scp target/ClusterTest1-1.0-jar-with-dependencies.jar ventura@bigdatalab.polito.it:dl4j

# Preprocessing
spark2-submit \
--master yarn \
--deploy-mode client \
--class preprocessing.MainPreprocessing \
--driver-memory 7G --executor-memory 7G --num-executors 4 \
--conf spark.executor.cores=4 --conf spark.driver.cores=4 \
dl4j/ClusterTest1-1.0-jar-with-dependencies.jar