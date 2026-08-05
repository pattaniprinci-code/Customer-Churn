import sys 
 
def getErrorDetails(error , error_Details:sys):
    _ , _ , tb_obj=error_Details.exc_info() #Traceback Object મેળવો
    Line_no=tb_obj.tb_lineno                 #Error ની line number

    file_name=tb_obj.tb_frame.f_code.co_filename     #Error આવેલી file નું નામ
    
    error_msg=f'erro occured in file name:-{file_name} ,error ocuured in line number :-{Line_no}'
    return error_msg


class customException(Exception):
    def __init__(self , error ,  error_msg : sys):
        super().__init__(error_msg)
        self.error_msg = getErrorDetails(error , error_msg)
        
        def __str__(self):
            return self.error_msg