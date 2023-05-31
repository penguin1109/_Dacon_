@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    epoch_loss = 0.0
    val_scores = 0.0
    running_crop_acc = 0.0
    running_disease_acc = 0.0
    running_final_acc = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, data in pbar:   
        eff_img = data['eff_img'].to(device)
        vit_img = data['vit_img'].to(device)
        csv_feats = data['csv_feature'].to(device)
        label = data['label'].to(device)   
        
        batch_size = eff_img.size(0)
        
        #y_pred  = model(img, csv_feats)
        pred_final = model(eff_img, vit_img,csv_feats)

        final_loss    = criterion(pred_final, label)
        final_acc = accuracy_function(label, pred_final)

        running_loss += (final_loss.item() * batch_size)

        dataset_size += batch_size
        running_final_acc += (final_acc.item() * batch_size)
        
        epoch_loss = running_loss / dataset_size
        val_final_scores = running_final_acc / dataset_size
        val_scores = val_final_scores
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        gpu_memory=f'{mem:0.2f} GB',
                         final_acc = f"{val_final_scores : 0.4f}")
    
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores
