import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob

class GetObjectLocationFromCamera_Class():
    """_summary_
    This class Store all function that can be used to find the location of object given pixel coordinates
    """
    def __init__(self):
        self.rMat = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]) \
                    @ np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]])
        self.tVec = np.zeros((3,), dtype="float32")
    
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
                                                                 angular_speed = None, forward_cor = 2, left_right_cor = None,
                                                                 left_sign = None, rMat = np.identity(3, dtype="float32"), 
                                                                 tVec = np.zeros((3,), dtype="float32"), disto = None):
        """_summary_
        Calculate the 3D coordinate from the position of camera when taking the first image
        
        Args:
            x1 ( np.array[1,2] ): image points from 1st image
            x2 ( np.array[1,2] ): image points from 2nd image
            cameraMat ( np.array[3,3] ): matrix related to intrinsic parameters of the camera
            speed ( float ): forword spped of the robot
            deltaT ( float ): time difference between 2 images
            angular_speed ( float , optional ) : angular speed of the robot counterclockwisely. Defaults to None.
            forward_cor ( int , optional): forward coordinate, 0 to x, 1 to y, 2 to z. Defaults to 2.
            left_right_cor (int , optional) : left and right coordinate, 0 to x, 1 to y, 2 to z. Defaults to None.
            left_sign (int , optional) : determine whether the left is +ive or -ive of left and right coordinate. Defaults to None.
            rMat ( np.array[3,3] , optional): Rotational matrix from world coordinates to camera coordinates. Defaults to np.identity(3, dtype="float32").
            tVec ( np.array[3,] , optional): Translation vector from  world coordinates to camera coordinates. Defaults to np.zeros((3,), dtype="float32").
            disto ( np.array[ 4/5/8/12/14 , optional ]) : distortion coefficients of the camera. Defaults to None.

        Returns:
            np.array[4,]: homogeneous 3d coordinates
        """
        if (angular_speed != None or left_right_cor != None or left_sign != None):
            assert angular_speed != None and left_right_cor != None and left_sign != None,\
                "angular_speed, left_righ_cor and left_sign must be specified at the same time"

        tVec = rMat @ tVec
        # 
        forward_dir_cam = np.matmul(rMat.T, np.array([int(forward_cor == i) for i in range(3)], dtype=int))
        forward_dir_cam = np.argmax(np.abs(forward_dir_cam))
        left_right_cor_cam = np.matmul(rMat.T, np.array([int(left_right_cor == i) for i in range(3)], dtype=int))
        left_right_cor_cam = np.argmax(np.abs(left_right_cor_cam))
        rMat2 = copy.deepcopy(rMat)
        tVec2 = copy.deepcopy(tVec)
        
        if angular_speed == None:
            # translational motion 
            tVec2[forward_dir_cam] -= forward_speed*deltaT
        else:
            tVec2[forward_dir_cam] -= forward_speed/angular_speed*np.sin(deltaT*angular_speed)
            tVec2[left_right_cor_cam] -= left_sign * forward_speed/angular_speed*(1 - np.cos(deltaT*angular_speed))
            rMat2 = np.matmul(rMat2, self.Rotate_Via_z_Axis_RightHand(deltaT*angular_speed))
        
        if not isinstance(disto,np.ndarray) :
            proj_mat1 = self.Calculate_Projective_Matrix(cameraMat,rMat,tVec)
            proj_mat2 = self.Calculate_Projective_Matrix(cameraMat,rMat2,tVec2)

            # solve the linear equations with Direct linear transformation (DLT)
            p_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, x1.T, x2.T)
            p_4d /= p_4d[3]
        else:
            x1_undistorted = cv2.undistortPoints(x1, cameraMatrix = cameraMat, distCoeffs = disto)
            x2_undistorted = cv2.undistortPoints(x2, cameraMatrix = cameraMat, distCoeffs = disto)
            proj_mat1 = self.Calculate_Projective_Matrix(np.identity(3, dtype="float32"),rMat,tVec)
            proj_mat2 = self.Calculate_Projective_Matrix(np.identity(3, dtype="float32"),rMat2,tVec2)
            p_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, x1_undistorted, x2_undistorted)
            p_4d /= p_4d[3]
            
        return p_4d.T
    
def DrawImage(img, bbox, relCor):
    """_summary_
    This function draws the bounding box of the object in image and 
    show the relative coordinates and distance of the object from the camera in meter.
        
    Args:
        img ( cv2.image ): image to be drawn
        bbox (list[4,]): bounding box of the object to be drawn
        relCor: relative coordinates from the camera
    """
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display image
    ax.imshow(img)

    # Add rectangle patch
    rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')   
    ax.add_patch(rect)

    relCor /= 100*10
    
    text = f'Rel. cord: ({relCor[0]:.4f} , {relCor[1]:.4f} , {relCor[2]:.4f})m\nDist: {(relCor[0]**2 + relCor[1]**2)**0.5:.4f}m'
    # add text to the rectangle
    ax.text(bbox[0], bbox[1], text, fontsize=12, color='r',ha='left')


    # Show plot
    plt.show()

    

if __name__ == "__main__":
    a = GetObjectLocationFromCamera_Class()