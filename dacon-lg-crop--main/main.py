import time
import copy
from copy import deepcopy
from collections import defaultdict

def run_training(model, optimizer, scheduler,device, num_epochs):
    # To automatically log gradients
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = model.state_dict()
    best_score  = -np.inf
    best_epoch = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        torch.cuda.empty_cache()

        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer,scheduler,
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, valid_loader, 
                                                         device=device, 
                                                         epoch=epoch)
        scheduler.step()
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Score'].append(val_scores)
        
        
        print(f'Valid Score: {val_scores:0.4f}')
        
        # deep copy the model
        if val_scores >= best_score:
            print(f"Valid Score Improved ({best_score:0.4f} ---> {val_scores:0.4f})")
            best_score   = val_scores
            best_epoch   = epoch
            best_model_wts = model.state_dict()
            #PATH = f"best_epoch-{fold:02d}.bin"
            PATH = f"/content/drive/MyDrive/dacon/LGfarm/ckpt/ViT_Mixed_fine2{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved")
            
      
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_score))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history
