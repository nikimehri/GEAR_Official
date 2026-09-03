import torch
from make_dataloaders import *
from tqdm import tqdm
import time

def run_exps(args, valset, testset, train_remain_loader, finetune=False, frozen=False):
    """Experimental (not part of the paper's reported results) "ensemble"
    approach: grafts the classifier head from a forget-optimized checkpoint
    (--good_forget) onto the backbone of a remain-optimized checkpoint
    (--good_remain), optionally fine-tunes the result on the remain set, and
    reports val/test forget+remain accuracy for the spliced model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    good_r = torch.load(args.good_remain, map_location=device)
    good_f = torch.load(args.good_forget, map_location=device)
    print(train_remain_loader.batch_size)

    if frozen:
        for param in good_r.parameters():
            param.requires_grad = False

        for param in good_r.module.classifier.parameters():
            param.requires_grad = True

    test_forget_loader, test_remain_loader = get_forget_loader(testset, args.forget_class)
    val_forget_loader, val_remain_loader = get_forget_loader(valset, args.forget_class)

    fc_from_good_f = good_f.module.classifier
    good_r.module.classifier = fc_from_good_f

    epochs = 0
    start=time.time()
    if finetune:

        optimizer = torch.optim.Adam(good_r.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        epochs = 5
        good_r.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for step, (batch_x, batch_y) in enumerate(tqdm(train_remain_loader)):

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = good_r(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_remain_loader)}')

    end=time.time()
    print(f'Time taken: {end-start}')

    val_remain_accuracy = evaluate(good_r, val_remain_loader,device)
    val_forget_accuracy = evaluate(good_r, val_forget_loader,device)

    print(f'Validation remain Set Accuracy with finetune:{finetune} for {epochs}: {val_remain_accuracy}')
    print(f'Validation forget Set Accuracy with finetune:{finetune} for {epochs}: {val_forget_accuracy}')

    test_remain_accuracy = evaluate(good_r, test_remain_loader, device)
    test_forget_accuracy = evaluate(good_r, test_forget_loader, device)
    print(f'TEST remain accuracy with finetune={finetune} for {epochs}: {test_remain_accuracy}')
    print(f'TEST forget accuracy with finetune={finetune} for {epochs}: {test_forget_accuracy}')


def evaluate(model, dataloader,device):
    """Plain top-1 accuracy over a full loader pass."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
