import numpy as np
import cv2

class GetObjectLocationFromCamera_Class():
    """_summary_
    This class Store all function that can be used to find the location of object given pixel coordinates
    """
    def Point2D_To_Point3D_with_z(self,point2D, rMat, tVec, cameraMat, z):
        """_summary_
        Function to convert given 2D image points to real-world 3D coordinates

        Args:
            point2D (list[?,2]): list of 2D image points
            rMat ( np.array[3,3] ): Rotational matrix from world coordinates to camera coordinates
            tVec ( np.array[3] ): Translation vector from  world coordinates to camera coordinates
            cameraMat ( np.array[3,3] ): intrinsic parameters of the camera
            z ( float ): coordinate in z-axis

        Returns:
            list[?,3] : list of real-world 3D coordinates

        """
        point3D = []
        point2D = (np.array(point2D,dtype="float32")).reshape(-1,2)
        num_Pts = point2D.shape[0]
        point2D_op = np.hstack((point2D, np.ones((num_Pts,1)) ))
        
        rMat_inv = np.linalg.inv(rMat)
        kMat_inv = np.linalg.inv(cameraMat)
        # print(num_Pts)
        for point in range(num_Pts):
            uvPoint = point2D_op[point, :]
            tempMat = np.matmul(rMat_inv, kMat_inv)
            tempMat1 = np.matmul(tempMat, uvPoint)
            tempMat2 = np.matmul(rMat_inv, tVec)
            s = (z + tempMat2[2])/tempMat1[2]
            p = tempMat1*s - tempMat2
            # print(p)
            point3D.append(p.tolist())
        return point3D
    
    def Calculate_Projective_Matrix(self,cameraMat, rMat, tVec):
        """_summary_
        Args:
            cameraMat ( np.array[3,3] ): matrix related to intrinsic parameters of the camera
            rMat ( np.array[3,3] ): Rotational matrix from world coordinates to camera coordinates
            tVec ( np.array[3,] ): Translation vector from  world coordinates to camera coordinates
        Returns:
             np.array[3,4]: Projective Matrix of the camera
        """
        extrinisicMat = np.zeros([3, 4])
        extrinisicMat[:,:3] = rMat
        extrinisicMat[:,3] = tVec
        projective_matrix = np.matmul(cameraMat, extrinisicMat)
        return projective_matrix
    
    def Rotate_Via_x_Axis_RightHand(self, angle):
        """_summary_

        Args:
            angle ( float ): rotation angle in radians ounterclockwisely via x axis 
                             following right hand rule

        Returns:
            np.array[3,3]: rotational matrix
        """
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    
    def Rotate_Via_y_Axis_RightHand(self, angle):
        """_summary_

        Args:
            angle ( float ): rotation angle in radians ounterclockwisely via y axis
                             following right hand rule

        Returns:
            np.array[3,3]: rotational matrix
        """

        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    
    def Rotate_Via_z_Axis_RightHand(self, angle):
        """_summary_

        Args:
            angle ( float ): rotation angle in radians ounterclockwisely via z axis
                             following right hand rule

        Returns:
            np.array[3,3]: rotational matrix
        """
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    
    def  Rotate_Via_i_Axis_RightHand(self, axis ,angle):
        """_summary_
        
        Args:
            axis ( int ): rotational axis, must be in 0, 1, 2
            angle ( float ): rotation angle in radians ounterclockwisely via i axis
                             following right hand rule      
        Returns:
            np.array[3,3]: rotational matrix
        
        """
        if axis == 0:
            return self.Rotate_Via_x_Axis_RightHand(angle)
        elif axis == 1:
            return self.Rotate_Via_y_Axis_RightHand(angle)
        elif axis == 2:
            return self.Rotate_Via_z_Axis_RightHand(angle)
        else:
            assert False, "axis must be 0, 1 or 2"
        
        

    def Calculate_3Dpts_2ImageFromSingleCamera_with_SpeedOfRobot(self, x1, x2, cameraMat, forward_speed, deltaT, 
                                                                 angular_speed = None, trans_cor = 2, left_right_cor = None, left_sign = None, rMat=None, tVec=None):
        """_summary_

        Args:
            x1 ( np.array[1,2] ): image points from 1st image
            x2 ( np.array[1,2] ): image points from 2nd image
            cameraMat ( np.array[3,3] ): matrix related to intrinsic parameters of the camera
            speed ( float ): forword spped of the robot
            deltaT ( float ): time difference between 2 images
            trans_cor ( int , optional): Translation coordinate, 0 to x, 1 to y, 2 to z. Defaults to 2.
            rMat ( np.array[3,3] , optional): Rotational matrix from world coordinates to camera coordinates. Defaults to None.
            tVec ( np.array[3,] , optional): Translation vector from  world coordinates to camera coordinates. Defaults to None.

        Returns:
            np.array[4,]: homogeneous 3d coordinates
        """
        if angular_speed != None or left_right_cor != None:
            assert angular_speed != None and left_right_cor != None ,"angular_speed and left_righ_cor must be specified at the same time"
        
            
        if rMat == None:
            rMat = np.identity(3, dtype="float32")
        if tVec == None:
            tVec = np.zeros((3,), dtype="float32")
        proj_mat1 = self.Calculate_Projective_Matrix(cameraMat,rMat,tVec)
        
        rMat2 = np.deepcopy(rMat)
        tVec2 = np.deepcopy(tVec)
        
        if angular_speed == None:
            # translational motion 
            tVec2[trans_cor] = forward_speed*deltaT
        else:
            tVec2[trans_cor] = forward_speed*deltaT*np.cos(deltaT*angular_speed)
            tVec2[left_right_cor] = left_sign * forward_speed*deltaT*np.sin(deltaT*angular_speed)
            rMat2 = np.matmul(rMat2, self.Rotate_Via_z_Axis_RightHand(deltaT*angular_speed))
        
        proj_mat2 = self.Calculate_Projective_Matrix(cameraMat,rMat2,tVec2)
        
        # solve the linear equations with Direct linear transformation (DLT)
        p = cv2.triangulatePoints(proj_mat1, proj_mat2, x1.T, x2.T)
        p /= p[3]
        return p.T

if __name__ == "__main__":
    a = GetObjectLocationFromCamera_Class()