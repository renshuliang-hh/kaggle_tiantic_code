import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

data_train = pd.read_csv('data/train.csv')
tf.reset_default_graph()
df=data_train
age_df=df[['Age','Fare','Parch','SibSp','Pclass']]
known_age = age_df[age_df.Age.notnull()].values  #知道的做训练集
unknown_age = age_df[age_df.Age.isnull()].values
test_x=known_age[:,1:]
test_y=known_age[:,0]

#param
num_steps=1000
batch_size=100
num_classes=81
num_features=4
num_trees=10
max_nodes=100000
X=tf.placeholder(tf.float32,shape=[None,num_features])
Y=tf.placeholder(tf.float32,shape=[None])
hparams=tensor_forest.ForestHParams(num_classes=num_classes,
                                    num_features=num_features,
                                    num_trees=num_trees,
                                    max_nodes=max_nodes).fill()

forest_graph=tensor_forest.RandomForestGraphs(params=hparams)
train_op=forest_graph.training_graph(X,Y)
loss_op=forest_graph.training_loss(X,Y)

infer_op, _, _=forest_graph.inference_graph(X)
correct_pre=tf.equal(tf.argmax(infer_op,1),tf.cast(Y,tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
init_vars=tf.group(tf.global_variables_initializer(),resources.initialize_resources(resources.shared_resources()))
sess=tf.Session()
sess.run(init_vars)
for i in range(num_steps):
    permutation = np.random.permutation(test_y.shape[0])
    batch_x=test_x[permutation,:][0:batch_size]
    batch_y=test_y[permutation][0:batch_size]
    _, l=sess.run([train_op,loss_op],feed_dict={X:batch_x,Y:batch_y})
    if i%50==0 or i==1:
        acc=sess.run(accuracy_op,feed_dict={X:test_x,Y:test_y})
        print('Step: %i Loss: %f Accuracy: %f' %(i,l,acc))
