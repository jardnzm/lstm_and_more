"""code just for illustration
"""
import YourDataLoader, YourDataset
from model import YourClass
from pathlib import Path
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim

lr_warm_up = True

def train(run_id: str, clean_data_root: Path, models_dir: Path, 
        save_every: int, backup_every: int, vis_every: int, force_restart: bool,
        visdom_server: str, no_visdom: bool, stepLR:bool):

    dataset = YourDataset(clean_data_root)
    loader = YourDataLoader(
        dataset,
        num_workers=8,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    model = YourClass(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    scheduler = None
    if stepLR:
        print("decay learning rate during trainig")
        scheduler = StepLR(optimizer=optimizer, step_size=150000, gamma=0.9)
    init_step = 1

    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")

    if lr_warm_up:
        # warm up the first 1/4 epochs
        warm_up_step = (len(loader) - init_step)//4
        warm_up_with_step_lr = lambda step: (step+1) / warm_up_step if step < warm_up_step \
            else 1
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_step_lr)
        
    model.train()
    
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Training loop
    print('total training steps: ')
    print(len(loader))
    for step, your_batch in enumerate(loader, init_step):
        # Forward pass
        inputs = torch.from_numpy(your_batch.data).to(device)
        sync(device)
        embeds = model(inputs)
        sync(device)
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)

        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()
        if scheduler:
          scheduler.step()

        if step % vis_every == 0:
            mesg = "{0}\tStep:{1} \tLoss:{2:.4f}\tEER:{3:.4f}\t".format(
                    time.ctime(), step,
                    loss.item(), eer)
            with open('exp/log','a') as f:
                f.write(mesg + ' \n')

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print(inputs.shape)
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)

        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
