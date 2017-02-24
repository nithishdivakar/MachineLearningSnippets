from datetime import datetime
import json

from keras.engine.training import GeneratorEnqueuer
import time


def train_iterations_on_generator(
        M,
        generator,
        iterations,
        snapshot_frequency,
        validation_frequency,
        val_generator, 
        val_samples,
        iteration_start=1
    ):
    
    loss_file = open('loss.txt',"a")
    pbar = tqdm(range(iteration_start,iterations+iteration_start))
    
    wait_time = 0.01
    enqueuer  = None
    try:
        # beg-- dafault values from API
        pickle_safe=False
        max_q_size = 10
        nb_worker = 10
        # end-- dafault values from API
        
        enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
        enqueuer.start(max_q_size=max_q_size, nb_worker=nb_worker)
        val_loss=None
        for iteration in pbar:
            pbar.set_description(str(iteration))
            
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(wait_time)
            x, y = generator_output
            
            loss = M.train_on_batch(x = x , y = y)
            
            DATUM = {}
            # print M.metrics_names,list(loss)
            
            for k,d in zip(M.metrics_names,loss):
                # print k,d
                DATUM['M_'+k] = float(d)
            
            
            if iteration % validation_frequency == 0 or iteration==iteration_start:
                val_loss = M.evaluate_generator(val_generator, val_samples)
                
            # print M.metrics_names,val_loss
            for k,d in zip(M.metrics_names,val_loss):
                # print k,d
                DATUM['M_val_'+k] = float(d)


            DATUM['time-stamp'] = datetime.now().strftime("%Y-%m-%d|%H:%M:%S:%f")
            DATUM['iteration']  = iteration
            loss_file.write('{}\n'.format(json.dumps(DATUM)))
            loss_file.flush()
            if iteration % snapshot_frequency == 0 :
                M.save("LOG/E1.{:s}.{:04d}.h5".format('---',iteration))
    finally:
        if enqueuer is not None:
            enqueuer.stop()
    loss_file.close()
