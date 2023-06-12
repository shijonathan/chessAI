import numpy as np
import torch

def context(dataset):
    """
    args:
    dataset: Dataset of tensors
   
    returns: 
    context_dataset: Dataset
    """
    context_dataset = []
        
    second_move = False
    # the starting board is the first element of the dataset
    starting_board = dataset[0][0]
    
    for i in range(len(dataset)):
        temp = []
        image, label = dataset[i]
        if torch.equal(image, starting_board):
            for j in range(3):
                temp.append(starting_board.clone())
                second_move = True
        elif second_move:
            for j in range(2):
                temp.append(starting_board.clone())
            temp.append(image)
            second_move = False
        else:
            for j in range(3):
                temp.append(dataset[i+j-2][0].clone())
                
        temp = torch.cat(temp)
        context_dataset.append(temp)
            
    context_dataset = torch.stack(context_dataset)
    new_dataset = torch.utils.data.TensorDataset(context_dataset, dataset[:][1])        
    
    return new_dataset
    
def worldview(dataset):
    """
    args:
    dataset: Dataset
   
    returns: 
    context_dataset: List
    """
    worldview_dataset = []
    
    for item in dataset:
        image = item[0]
        label = item[1]
        chessboard_black = torch.reshape(image, (8, 8, 12))
        chessboard_black = chessboard_black.flip(0)
        chessboard_black = chessboard_black.flip(2)
        chessboard_black = torch.reshape(chessboard_black, (768,))
        
        temp = torch.cat((image, chessboard_black))
        
        worldview_dataset.append((temp, label))
    
    return worldview_dataset