/* script reads data_train.csv
 * calculates mode of categorical variables (only columns with missing values)
 * calculates means of numeric variables (only columns with missing values)
 * modes and means for missing value columns are calculated seperately for both classes
 * above values are used to fill missing values
 * categorical variables are encoded using a vector (if a variable has 4 levels then it is encoded using vector of length 4)
 * Due to uneven classes, one class is trained with 26 equal partitions of zero class...resulting in 26 models
 * reads data_test.csv
 * adds missing values assuming one class and then assuming zero class
 * predictions are made for both cases using 26 models and an average is taken
 */

/* read data_train.csv */

val df = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("/praveen/data_train.csv")

case class InputOutput(id: Option[Int], num1: Option[Int], num2: Option[Int], num3: Option[Int], num4: Option[Int], num5: Option[Int], num6: Option[Int], num7: Option[Int], num8: Option[Int], num9: Option[Int], num10: Option[Int], num11: Option[Int], num12: Option[Int], num13: Option[Int], num14: Option[Int], num15: Option[Int], num16: Option[Double], num17: Option[Double], num18: Option[Double], num19: Option[Double], num20: Option[Double], num21: Option[Double], num22: Option[Double], num23: Option[Double], der1: Option[Double], der2: Option[Double], der3: Option[Double], der4: Option[Int], der5: Option[Int], der6: Option[Int], der7: Option[Int], der8: Option[Int], der9: Option[Int], der10: Option[Int], der11: Option[Int], der12: Option[Int], der13: Option[Int], der14: Option[Int], der15: Option[Int], der16: Option[Int], der17: Option[Int], der18: Option[Int], der19: Option[Int], cat1: Option[Double], cat2: Option[Double], cat3: Option[Double], cat4: Option[Double], cat5: Option[Double], cat6: Option[Double], cat7: Option[Int], cat8: Option[Double], cat9: Option[Int], cat10: Option[Double], cat11: Option[Int], cat12: Option[Double], cat13: Option[Int],cat14: Option[Int], target: Option[Int])

/* partition data to take advantage of 8 cores */

val repart = df.repartition(8)
val ds = repart.as[InputOutput]
val dsones = ds.filter(a => a.target.get == 1)
val dszeros = ds.filter(a => a.target.get == 0)
val dsonesummary = dsones.describe().collect()
val dszerosummary = dszeros.describe().collect()

/* calculate mode and mean of missing values for one class*/

val cat1mode = dsones.map(a => (a.cat1)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat2mode = dsones.map(a => (a.cat2)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat3mode = dsones.map(a => (a.cat3)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat4mode = dsones.map(a => (a.cat4)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat5mode = dsones.map(a => (a.cat5)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat6mode = dsones.map(a => (a.cat6)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat8mode = dsones.map(a => (a.cat8)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat10mode = dsones.map(a => (a.cat10)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat12mode = dsones.map(a => (a.cat12)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val num18mean = dsonesummary(1).getString(20).toDouble
val num19mean = dsonesummary(1).getString(21).toDouble
val num20mean = dsonesummary(1).getString(22).toDouble
val num22mean = dsonesummary(1).getString(24).toDouble

/* fill missing values for one class */

val dsonenona = dsones.map (a => (a.target.get, org.apache.spark.ml.linalg.Vectors.dense(a.num1.get,a.num2.get,a.num3.get, a.num4.get,a.num5.get, a.num6.get,a.num7.get,a.num8.get,a.num9.get, a.num10.get, a.num11.get, a.num12.get, a.num13.get,a.num14.get,a.num15.get,a.num16.get, a.num17.get,a.num18.getOrElse(num18mean),a.num19.getOrElse(num19mean),a.num20.getOrElse(num20mean),a.num21.get,a.num22.getOrElse(num22mean),a.num23.get,a.der1.get,a.der2.get,a.der3.get,a.der4.get,a.der5.get,a.der6.get,a.der7.get,a.der8.get,a.der9.get,a.der10.get,a.der11.get,a.der12.get,a.der13.get,a.der14.get,a.der15.get,a.der16.get,a.der17.get,a.der18.get,a.der19.get,a.cat1.getOrElse(cat1mode),a.cat2.getOrElse(cat2mode),a.cat3.getOrElse(cat3mode),a.cat4.getOrElse(cat4mode), a.cat5.getOrElse(cat5mode),a.cat6.getOrElse(cat6mode),a.cat7.get,a.cat8.getOrElse(cat8mode), a.cat9.get,a.cat10.getOrElse(cat10mode), a.cat11.get, a.cat12.getOrElse(cat12mode),a.cat13.get, a.cat14.get),a.id.get))

/* calculate mode and mean for zero class */

val cat1mode = dszeros.map(a => (a.cat1)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat2mode = dszeros.map(a => (a.cat2)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat3mode = dszeros.map(a => (a.cat3)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat4mode = dszeros.map(a => (a.cat4)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat5mode = dszeros.map(a => (a.cat5)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat6mode = dszeros.map(a => (a.cat6)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat8mode = dszeros.map(a => (a.cat8)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat10mode = dszeros.map(a => (a.cat10)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat12mode = dszeros.map(a => (a.cat12)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val num18mean = dszerosummary(1).getString(20).toDouble
val num19mean = dszerosummary(1).getString(21).toDouble
val num20mean = dszerosummary(1).getString(22).toDouble
val num22mean = dszerosummary(1).getString(24).toDouble

/* fill missing values for zero class */

val dszeronona = dszeros.map (a => (a.target.get, org.apache.spark.ml.linalg.Vectors.dense(a.num1.get,a.num2.get,a.num3.get, a.num4.get,a.num5.get, a.num6.get,a.num7.get,a.num8.get,a.num9.get, a.num10.get, a.num11.get, a.num12.get, a.num13.get,a.num14.get,a.num15.get,a.num16.get, a.num17.get,a.num18.getOrElse(num18mean),a.num19.getOrElse(num19mean),a.num20.getOrElse(num20mean),a.num21.get,a.num22.getOrElse(num22mean),a.num23.get,a.der1.get,a.der2.get,a.der3.get,a.der4.get,a.der5.get,a.der6.get,a.der7.get,a.der8.get,a.der9.get,a.der10.get,a.der11.get,a.der12.get,a.der13.get,a.der14.get,a.der15.get,a.der16.get,a.der17.get,a.der18.get,a.der19.get,a.cat1.getOrElse(cat1mode),a.cat2.getOrElse(cat2mode),a.cat3.getOrElse(cat3mode),a.cat4.getOrElse(cat4mode), a.cat5.getOrElse(cat5mode),a.cat6.getOrElse(cat6mode),a.cat7.get,a.cat8.getOrElse(cat8mode), a.cat9.get,a.cat10.getOrElse(cat10mode), a.cat11.get, a.cat12.getOrElse(cat12mode),a.cat13.get, a.cat14.get),a.id.get))

/*partition zero class*/

val g = for (i <- 1 to 26) yield 1.0
val rsplits = g.toArray
val splitdszeronona = dszeronona.randomSplit(rsplits,101)
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg._
val lr = new LogisticRegression().setLabelCol("_1").setFeaturesCol("_2")

/* function to encode categorical variables */

def cattodouble (cats: Array[Double]): Array[Double] = {
    var retArray: Array[Double] = Array()
    val maxcats: Array[Int] = Array(4,1,6,11,1,1,9,1,17,1,1,4,2,104)
    var g = for(i <- 1 to 5) yield 0.0
    var ga = g.toArray
    for (i <- 1 to maxcats.size) {
       g = for(i <- 0 to (maxcats(i-1) + 1)) yield 0.0
       ga = g.toArray
       ga(cats(i - 1).toInt) = 1.0
       retArray = retArray.union(ga)
    }
    retArray
}

/* create 26 models */

var lrModels: Array[org.apache.spark.ml.classification.LogisticRegressionModel] = Array()
var trainarray: Array[org.apache.spark.sql.Dataset[(Int, org.apache.spark.ml.linalg.Vector, Int)]] = Array()
var mtrainarray: Array[org.apache.spark.sql.Dataset[(Int, org.apache.spark.ml.linalg.Vector, Int)]] = Array()
for (j <- 1 to splitdszeronona.size) {
    trainarray = trainarray.union(Array(splitdszeronona(j - 1).union(dsonenona)))

    /* encode categorical variables */

    mtrainarray = mtrainarray.union(Array(trainarray(j - 1).map(a => (a._1, org.apache.spark.ml.linalg.Vectors.dense(a._2.toArray.take(42).union(cattodouble(a._2.toArray.takeRight(14)))),a._3))))
    mtrainarray(j - 1).persist
    lrModels = lrModels.union(Array(lr.fit(mtrainarray(j - 1))))
}

case class pid(probability: Vector,_3: Int,_1: Int)

/* predict using all the models */

def predictUsingCombinedModel(dataset: org.apache.spark.sql.Dataset[(Int, org.apache.spark.ml.linalg.Vector, Int)],
                              models:  Array[org.apache.spark.ml.classification.LogisticRegressionModel]): org.apache.spark.sql.Dataset[(Int, (Double, Double, Int, Int))] = {
    var results: org.apache.spark.sql.Dataset[(Double, Double, Int, Int)] = spark.emptyDataset[(Double, Double, Int, Int)]
       for (i <- 1 to models.size) {
           val p = models(i - 1).transform(dataset)
           val p1 = p.select("probability","_3", "_1")
           val p2 = p1.as[pid]
           val p3 = p2.map(a => (a.probability(0), a.probability(1), a._3, a._1))
           results = results.union(p3)
       }
    results.groupByKey(a => a._3).reduceGroups((a,b) => (a._1 + b._1, a._2 + b._2, a._3, a._4))
}

/* read test data */

val dft = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("/praveen/data_test.csv")
val dftsummary = dft.describe()
dftsummary.show
dftsummary.first()
case class TInputOutput(id: Option[Int], num1: Option[Int], num2: Option[Int], num3: Option[Int], num4: Option[Int], num5: Option[Int], num6: Option[Int], num7: Option[Int], num8: Option[Int], num9: Option[Int], num10: Option[Int], num11: Option[Int], num12: Option[Int], num13: Option[Int], num14: Option[Int], num15: Option[Int], num16: Option[Double], num17: Option[Double], num18: Option[Double], num19: Option[Double], num20: Option[Double], num21: Option[Double], num22: Option[Double], num23: Option[Double], der1: Option[Double], der2: Option[Double], der3: Option[Double], der4: Option[Int], der5: Option[Int], der6: Option[Int], der7: Option[Int], der8: Option[Int], der9: Option[Int], der10: Option[Int], der11: Option[Int], der12: Option[Int], der13: Option[Int], der14: Option[Int], der15: Option[Int], der16: Option[Int], der17: Option[Int], der18: Option[Int], der19: Option[Int], cat1: Option[Double], cat2: Option[Double], cat3: Option[Double], cat4: Option[Double], cat5: Option[Double], cat6: Option[Double], cat7: Option[Int], cat8: Option[Double], cat9: Option[Int], cat10: Option[Double], cat11: Option[Int], cat12: Option[Double], cat13: Option[Int],cat14: Option[Int])

/* partition to take advantage of 8 cores */

val trepart = dft.repartition(8)
val dst = trepart.as[TInputOutput]

/* assuming one class fill the missing values  and predict using 26 models*/

val cat1mode = dsones.map(a => (a.cat1)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat2mode = dsones.map(a => (a.cat2)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat3mode = dsones.map(a => (a.cat3)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat4mode = dsones.map(a => (a.cat4)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat5mode = dsones.map(a => (a.cat5)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat6mode = dsones.map(a => (a.cat6)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat8mode = dsones.map(a => (a.cat8)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat10mode = dsones.map(a => (a.cat10)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat12mode = dsones.map(a => (a.cat12)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val num18mean = dsonesummary(1).getString(20).toDouble
val num19mean = dsonesummary(1).getString(21).toDouble
val num20mean = dsonesummary(1).getString(22).toDouble
val num22mean = dsonesummary(1).getString(24).toDouble
val dstonenona = dst.map (a => (1, org.apache.spark.ml.linalg.Vectors.dense(a.num1.get,a.num2.get,a.num3.get, a.num4.get,a.num5.get, a.num6.get,a.num7.get,a.num8.get,a.num9.get, a.num10.get, a.num11.get, a.num12.get, a.num13.get,a.num14.get,a.num15.get,a.num16.get, a.num17.get,a.num18.getOrElse(num18mean),a.num19.getOrElse(num19mean),a.num20.getOrElse(num20mean),a.num21.get,a.num22.getOrElse(num22mean),a.num23.get,a.der1.get,a.der2.get,a.der3.get,a.der4.get,a.der5.get,a.der6.get,a.der7.get,a.der8.get,a.der9.get,a.der10.get,a.der11.get,a.der12.get,a.der13.get,a.der14.get,a.der15.get,a.der16.get,a.der17.get,a.der18.get,a.der19.get,a.cat1.getOrElse(cat1mode),a.cat2.getOrElse(cat2mode),a.cat3.getOrElse(cat3mode),a.cat4.getOrElse(cat4mode), a.cat5.getOrElse(cat5mode),a.cat6.getOrElse(cat6mode),a.cat7.get,a.cat8.getOrElse(cat8mode), a.cat9.get,a.cat10.getOrElse(cat10mode), a.cat11.get, a.cat12.getOrElse(cat12mode),a.cat13.get, a.cat14.get),a.id.get))
val dstonemtrain = dstonenona.map(a => (a._1, org.apache.spark.ml.linalg.Vectors.dense(a._2.toArray.take(42).union(cattodouble(a._2.toArray.takeRight(14)))),a._3))
val predtone = predictUsingCombinedModel(dstonemtrain, lrModels)

/* assuming zero class fill the missing values  and predict using 26 models*/

val cat1mode = dszeros.map(a => (a.cat1)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat2mode = dszeros.map(a => (a.cat2)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat3mode = dszeros.map(a => (a.cat3)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat4mode = dszeros.map(a => (a.cat4)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat5mode = dszeros.map(a => (a.cat5)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat6mode = dszeros.map(a => (a.cat6)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat8mode = dszeros.map(a => (a.cat8)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat10mode = dszeros.map(a => (a.cat10)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val cat12mode = dszeros.map(a => (a.cat12)).filter(a => a != None).map(a => a.get).groupBy("value").count().collect().map(a => (a.getDouble(0), a.getLong(1))).maxBy(_._2)._1
val num18mean = dszerosummary(1).getString(20).toDouble
val num19mean = dszerosummary(1).getString(21).toDouble
val num20mean = dszerosummary(1).getString(22).toDouble
val num22mean = dszerosummary(1).getString(24).toDouble
val dstzeronona = dst.map (a => (1, org.apache.spark.ml.linalg.Vectors.dense(a.num1.get,a.num2.get,a.num3.get, a.num4.get,a.num5.get, a.num6.get,a.num7.get,a.num8.get,a.num9.get, a.num10.get, a.num11.get, a.num12.get, a.num13.get,a.num14.get,a.num15.get,a.num16.get, a.num17.get,a.num18.getOrElse(num18mean),a.num19.getOrElse(num19mean),a.num20.getOrElse(num20mean),a.num21.get,a.num22.getOrElse(num22mean),a.num23.get,a.der1.get,a.der2.get,a.der3.get,a.der4.get,a.der5.get,a.der6.get,a.der7.get,a.der8.get,a.der9.get,a.der10.get,a.der11.get,a.der12.get,a.der13.get,a.der14.get,a.der15.get,a.der16.get,a.der17.get,a.der18.get,a.der19.get,a.cat1.getOrElse(cat1mode),a.cat2.getOrElse(cat2mode),a.cat3.getOrElse(cat3mode),a.cat4.getOrElse(cat4mode), a.cat5.getOrElse(cat5mode),a.cat6.getOrElse(cat6mode),a.cat7.get,a.cat8.getOrElse(cat8mode), a.cat9.get,a.cat10.getOrElse(cat10mode), a.cat11.get, a.cat12.getOrElse(cat12mode),a.cat13.get, a.cat14.get),a.id.get))
val dstzeromtrain = dstzeronona.map(a => (a._1, org.apache.spark.ml.linalg.Vectors.dense(a._2.toArray.take(42).union(cattodouble(a._2.toArray.takeRight(14)))),a._3))
val predtzero = predictUsingCombinedModel(dstzeromtrain, lrModels)

/* combine one class and zero class predictions
 * write predictions to file
 */

val predt = predtzero.union(predtone).groupByKey(a => a._1).reduceGroups((a,b) => (a._1, (a._2._1 + b._2._1, a._2._2 + b._2._2, a._2._3, a._2._4)))
val fpredt = predt.map(a => (a._1, a._2._2._2/26.0))
val cfpredt = fpredt.collect()
val finalpred = cfpredt.map(a => (a._1, a._2/2.0))
import java.io._
val writer = new PrintWriter(new File("/home/praveen/kaggle/quartic.ai/ds_data/output.txt"))
writer.write("id,target(probability of 1)")
finalpred.foreach(a => { writer.write("\n" + a._1.toString + "," + a._2.toString) })
writer.close()
