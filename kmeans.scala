import scala.collection.immutable.Vector

def distance(p: Vector[Double], q: Vector[Double]): Double = {
  require(p.length == q.length, "Points must have equal dimensions")
  math.sqrt(p.zip(q).map {
    case (a, b) => (a - b) * (a - b)
  }.sum)
}

def clostestpoint(
    q: Vector[Double],
    candidates: Array[Vector[Double]]
): Vector[Double] = {
  require(candidates.nonEmpty, "Candidates cannot be empty")
  candidates.minBy(p => distance(q, p))
}

def add_vec(
    v1: Vector[Double],
    v2: Vector[Double]
): Vector[Double] = {
  require(v1.length == v2.length, "Vectors must have equal dimensions")
  v1.zip(v2).map {
    case (a, b) => a + b
  }
}

def average(cluster: Iterable[Vector[Double]]): Vector[Double] = {
  require(cluster.nonEmpty, "Cluster cannot be empty")
  val total = cluster.reduce((a, b) => add_vec(a, b))
  total.map(_ / cluster.size.toDouble)
}

////////////////////////
// Main Program Below //
////////////////////////

val dataPath =
  "file:///home/bitnami/Desktop/Data-Transfer-2-VM/clustering_dataset.txt"

val points = sc.textFile(dataPath).filter(line => line.trim.nonEmpty).map(line => line.trim.split("\\s+").map(_.toDouble).toVector).cache()

val k = 3
val maxIterations = 100
val tolerance = 0.000001

require(points.count() >= k, "Not enough data points")

var centroids = points.distinct().takeSample(false, k, 42L)
require(centroids.length == k, "Not enough distinct points")

var iteration = 0
var movement = Double.PositiveInfinity

while (iteration < maxIterations && movement > tolerance) {
  val currentCentroids = centroids

  val updated = points
    .map(point => (clostestpoint(point, currentCentroids), point))
    .groupByKey()
    .map {
      case (center, cluster) => (center, average(cluster))
    }
    .collect()
    .toMap

  val nextCentroids = currentCentroids.map { center =>
    updated.getOrElse(center, center)
  }

  movement = currentCentroids.zip(nextCentroids)
    .map {
      case (oldCenter, newCenter) => distance(oldCenter, newCenter)
    }
    .max

  centroids = nextCentroids
  iteration += 1

  println("Iteration " + iteration + ", movement = " + movement)
}

println("Final cluster centroids:")
centroids.zipWithIndex.foreach {
  case (center, index) =>
    println(
      "Centroid " + (index + 1) + ": " +
      center.mkString("(", ", ", ")")
    )
}

if (movement > tolerance) {
  println("Warning: maximum iterations reached before convergence.")
}

points.unpersist()
