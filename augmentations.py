import numpy as np

def context(dataset):
    """
    args:
    dataset: List
   
    returns: 
    context_dataset: List
    """
    context_dataset = []
    memory = 2
        
    # the starting board is the first element of the dataset
    starting_board = dataset[0][0]
    
    for i in range(len(dataset)):
        temp = []
        image = dataset[i][0]
        label = dataset[i][1]
        if np.array_equal(image, starting_board) or memory < 2:
            for j in range(memory):
                temp.append(starting_board)
            memory -= 1
            idx = 1
            while len(temp) < 3:
                temp.append(dataset[i - idx][0])
                idx -= 1
                
            if memory <= 0:
                memory = 2
            
        else:
            for j in range(3):
                temp.append(dataset[i+j-2][0])
        
        context_dataset.append((np.array(temp).flatten(), label))
            
    return context_dataset
    
def worldview(dataset):
    """
    args:
    dataset: List or np.array
   
    returns: 
    context_dataset: List
    """
    worldview_dataset = []
    
    for image, label in dataset:
        chessboard_black = np.reshape(image, (8, 8, 12))
        chessboard_black = np.flip(chessboard_black, axis=0)
        chessboard_black[:, :, :6], chessboard_black[:, :, 6:] = chessboard_black[:, :, 6:], chessboard_black[:, :, :6]
        onehot_black = chessboard_black.flatten()
        
        temp = np.append(image, onehot_black)
        
        worldview_dataset.append((temp, label))
    
    return worldview_dataset