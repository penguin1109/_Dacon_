 # Training Function
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, data in pbar:   
        vit_img = data['vit_img'].to(device)
        eff_img = data['eff_img'].to(device)
        csv_feats = data['csv_feature'].to(device)
        label = data['label'].to(device)
            
        batch_size = eff_img.size(0)
        optimizer.zero_grad()

        with amp.autocast(enabled=True):
            pred_final = model(eff_img,vit_img, csv_feats)
            final_loss = criterion(pred_final, label)
        
        """
        scaler.scale(crop_loss).backward()
        scaler.step(optimizer1)
        scaler.update()

        scaler.scale(disease_loss).backward()
        scaler.step(optimizer2)
        scaler.update()
        """
        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # zero the parameter gradients
        """
        if scheduler is not None:
          scheduler.step(epoch +step / len(dataloader))
        """
        
        running_loss += (final_loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        
        pbar.set_postfix(
            train_loss=f'{epoch_loss:0.4f}',
                        gpu_memory=f'{mem:0.2f} GB',)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss
