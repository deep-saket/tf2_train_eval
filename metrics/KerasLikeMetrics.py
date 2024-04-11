###########################################################################################
# Parent class of all the metrics                                                         #                                               
#       * inspired from keras metrics                                                     #
#                                                                                         #
# Developed by - Saket Mohanty                                                            #
###########################################################################################


class KerasLikeMetrics:
    '''
    This is an abstract class, derived class of this class 
    supposed to work like keras metrics

    override all the 3 method below to use this class
    '''
        
    @abstractmethod
    def update_state(self, y_pred, y_true):
        '''
        Add example to metric evaluation list

        Args --
            y_pred -- predictions
            y_true -- labels
        '''
        pass

    @abstractmethod
    def result(self):
        '''
        Return computed metric result on the examples
        present in the evaluation list
        '''
        pass

    @abstractmethod
    def reset_state(self):
        '''
        Resets evaluation list to 0
        '''
        pass