from EmotionDataset import EmotionDataset
from Transformer import Transformer
from HalveTransformer import HalveTransformer
import termcolor
import random
import pprint
import numpy
import torch
import click
import tqdm
import json
import time

models={"Transformer":Transformer, "HalveTransformer":HalveTransformer}
MODELS=list(models.keys())

@click.command()
@click.option("--epochs"           , "epochs"           , type=int                  , default=10            , help="Number of epochs to train for" )
@click.option("--train-batch-size" , "train_batch_size" , type=int                  , default=128           , help="Batch size for training"       )
@click.option("--valid-batch-size" , "valid_batch_size" , type=int                  , default=128           , help="Batch size for validation"     )
@click.option("--test-batch-size"  , "test_batch_size"  , type=int                  , default=128           , help="Batch size for testing"        )
@click.option("--learning-rate"    , "learning_rate"    , type=float                , default=0.001         , help="Learning rate for training"    )
@click.option("--device"           , "device"           , type=str                  , default="cuda:0"      , help="Device to train on"            )
@click.option("--seed"             , "seed"             , type=int                  , default=42            , help="Seed for reproducibility"      )
@click.option("--dir"              , "dir"              , type=click.Path()         , default="."           , help="Directory to save model"       )
@click.option("--model"            , "model"            , type=click.Choice(MODELS) , default="Transformer" , help="Model to train"                )
def run(
        seed             = 42            ,
        epochs           = 10            ,
        train_batch_size = 128           ,
        valid_batch_size = 128           ,
        test_batch_size  = 128           ,
        learning_rate    = 0.001         ,
        dir              = "."           ,
        model            = "Transformer" ,
        device           = "cuda:0"      ,
    ):

    # reset memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # seed everything
    torch.manual_seed      (seed)
    torch.cuda.manual_seed (seed)
    numpy.random.seed      (seed)
    random.seed            (seed)

    # print configuration
    print(" config ".center(80, "="))
    pprint.pprint(locals())
    print("=".center(80, "="))

    open(f"{dir}/config.json", "w").write(json.dumps(locals()))
    trainfile = open(f"{dir}/train.jsonl", "w")
    validfile = open(f"{dir}/valid.jsonl", "w")
    testfile  = open(f"{dir}/test.jsonl" , "w")

    trainloader = torch.utils.data.DataLoader(EmotionDataset(split="train"     ), batch_size=train_batch_size, shuffle=True, collate_fn=EmotionDataset.collate_fn)
    validloader = torch.utils.data.DataLoader(EmotionDataset(split="validation"), batch_size=valid_batch_size, shuffle=True, collate_fn=EmotionDataset.collate_fn)
    testloader  = torch.utils.data.DataLoader(EmotionDataset(split="test"      ), batch_size= test_batch_size, shuffle=True, collate_fn=EmotionDataset.collate_fn)
    model       = models[model]().to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion   = torch.nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(epochs):
        model.train()
        total,tpfn,cumloss = 0,0,0
        for step, batch in enumerate(bar := tqdm.tqdm(trainloader)):
            starttime = time.time()
            optimizer.zero_grad()
            y_pred = model(batch["src"].to(device))
            loss = criterion(y_pred, batch["tgt"].to(device))
            loss.backward()
            optimizer.step()
            endtime = time.time()
            cumloss = cumloss + loss.item()
            total,tpfn = batch["tgt"].size(0) + total, tpfn + (y_pred.argmax(1) == batch["tgt"].to(device)).long().sum().item()
            bar.set_description(termcolor.colored(f"[train] epoch {epoch: <4}, step:{step: <4}, acc: {tpfn/total:.4f}, lss: {loss.item():.4f}", "white"))
            trainfile.write(json.dumps({
                "epoch"    : epoch,
                "step"     : step,
                "accuracy" : tpfn/total,
                "loss"     : cumloss/len(trainloader),
                "gpumem"   : torch.cuda.max_memory_allocated() / 1024**2,
                "time"     : endtime-starttime,
            }) + "\n")

        model.eval()
        total,tpfn,cumloss = 0,0,0
        for step, batch in enumerate(bar := tqdm.tqdm(validloader)):
            starttime = time.time()
            y_pred = model(batch["src"].to(device))
            loss = criterion(y_pred, batch["tgt"].to(device))
            endtime = time.time()
            cumloss = cumloss + loss.item()
            total,tpfn = batch["tgt"].size(0) + total, tpfn + (y_pred.argmax(1) == batch["tgt"].to(device)).long().sum().item()
            bar.set_description(termcolor.colored(f"[valid] epoch {epoch: <4}, step:{step: <4}, acc: {tpfn/total:.4f}, lss: {loss.item():.4f}","cyan"))
        validfile.write(json.dumps({
                "epoch"    : epoch,
                "step"     : step,
                "accuracy" : tpfn/total,
                "loss"     : cumloss/len(trainloader),
                "time"     : endtime-starttime,
                "gpumem"   : torch.cuda.max_memory_allocated() / 1024**2,
            }) + "\n")

    model.eval()
    total,tpfn,cumloss = 0,0,0
    starttime = time.time()
    for step, batch in enumerate(bar := tqdm.tqdm(testloader)):
        y_pred = model(batch["src"].to(device))
        loss = criterion(y_pred, batch["tgt"].to(device))
        cumloss = cumloss + loss.item()
        total,tpfn = batch["tgt"].size(0) + total, tpfn + (y_pred.argmax(1) == batch["tgt"].to(device)).long().sum().item()
        bar.set_description(termcolor.colored(f"[test ] step:{step: <4}, acc: {tpfn/total:.4f}, lss: {loss.item():.4f}", "green"))
    endtime = time.time()
    testfile.write(json.dumps({
                "accuracy" : tpfn/total,
                "loss"     : cumloss/len(trainloader),
                "gpumem"   : torch.cuda.max_memory_allocated() / 1024**2,
                "time"     : endtime-starttime,
            }) + "\n")

    # close files
    trainfile.close()
    validfile.close()
    testfile.close()


if __name__ == "__main__":
    run()
