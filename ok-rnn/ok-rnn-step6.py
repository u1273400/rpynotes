# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0
