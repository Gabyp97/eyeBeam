import numpy as np
import pandas as pd
import math
def load_data(fileName,*relevant_columns):
    #this function load excel tracker file, and pull out the relevant columns (is definded in "relevant_columns" argument)
    #for transformation of mask's gaze vector (from mask coordinate system) to Unity coordinate system.
    #return the relevant data in dataFrame
    data = pd.read_csv(fileName)
    relevant_data = data[list(relevant_columns)]
    return relevant_data

def split_DataFrame(dataFrame,rotation_col,right_eye_col,left_col):
    rotation_data = dataFrame[list(rotation_col)]
    rightEye_gaze = dataFrame[list(right_eye_col)]
    leftEye_gaze = dataFrame[list(left_col)]

    return rotation_data,rightEye_gaze,leftEye_gaze
def create_rotation_mat(rotationData,euler_z,euler_x,euler_y):
    #create rotation matrix
    #input: row from dataFrame
    teta_y = math.radians(rotationData[euler_y])
    teta_x = math.radians(rotationData[euler_x])
    teta_z = math.radians(rotationData[euler_z])
    rotation_mat_y = np.array([[math.cos(teta_y),0,math.sin(teta_y)],[0,1,0],[-(math.sin(teta_y)),0,math.cos(teta_y)]])
    rotation_mat_x = np.array([[1,0,0],[0,math.cos(teta_x),-(math.sin(teta_x))],[0,math.sin(teta_x),math.cos(teta_x)]])
    rotation_mat_z = np.array([[math.cos(teta_z),-(math.sin(teta_z)),0],[math.sin(teta_z),math.cos(teta_z),0],[0,0,1]])
    return rotation_mat_y,rotation_mat_x,rotation_mat_z



def mult_matrix(z,x,y,vec):
    # the order of the arguments is importent
    vec[0] = -vec[0]
    zmat_xmat_product = np.matmul(z,x)
    gaze_ymat_product = np.matmul(y,vec)
    gaze_vec_unity =  np.matmul(zmat_xmat_product,gaze_ymat_product)
    return gaze_vec_unity


def from_mask_to_unity(dataFrame,rotation_col,right_eye_col,left_eye_col):
    #always write the column's name in this order : x,y,z
    rotationData,rightEye_gaze,leftEye_gaze = split_DataFrame(dataFrame,rotation_col,right_eye_col,left_eye_col)
    right_gaze_unity_mat = []
    left_gaze_unity_mat = []
    rotationData = rotationData.set_axis(['x_euler','y_euler','z_euler'],axis=1, inplace=False)
    for i in range(len(rotationData)):
        rotation_mat_y, rotation_mat_x, rotation_mat_z = create_rotation_mat(rotationData.iloc[[i]],'x_euler','y_euler','z_euler')
        right_gaze_vec_mask = np.array(rightEye_gaze.iloc[i])
        left_gaze_vec_mask = np.array(leftEye_gaze.iloc[i])
        right_gaze_vec_unity = mult_matrix(rotation_mat_z,rotation_mat_x,rotation_mat_y,right_gaze_vec_mask)
        left_gaze_vec_unity = mult_matrix(rotation_mat_z,rotation_mat_x,rotation_mat_y,left_gaze_vec_mask)
        right_gaze_unity_mat.append(right_gaze_vec_unity)
        left_gaze_unity_mat.append(left_gaze_vec_unity)
    r_gaze_dataframe = pd.DataFrame(right_gaze_unity_mat,columns=['x','y','z'])
    l_gaze_dataframe = pd.DataFrame(left_gaze_unity_mat,columns=['x','y','z'])
    return r_gaze_dataframe,l_gaze_dataframe


def beam_hit(headset_position,vector,delta,relevant_depth,x_nameP,y_nameP,z_nameP,x_nameV,y_nameV,z_nameV):
    distance_between_eye_releventDepth = -(headset_position[z_nameP] - relevant_depth)
    t = np.array(distance_between_eye_releventDepth/vector[z_nameV])
    a = np.array(headset_position[x_nameP]+delta + t*vector[x_nameV])
    b = np.array(headset_position[y_nameP]+t*vector[y_nameV])
    c = np.array(headset_position[z_nameP]+ t*vector[z_nameV])
    new_vec = np.array([a , b , c])
    return new_vec

####check######
#rot_data = pd.DataFrame(data = {'x':[356.26],'y':[-264.84],'z':[1.61]})
#euler_mat_y,euler_mat_x,euler_mat_z = create_rotation_mat(rot_data,'z','x','y')
#vec1 = np.array([-0.326,-0.108,0.92])
#ans = mult_matrix(euler_mat_z,euler_mat_x,euler_mat_y,vec1)
#ans = pd.DataFrame(data=np.reshape(ans,[1,3]),columns=['x', 'y', 'z'])
#pos_data = pd.DataFrame(data={'x_head': [-8.53], 'y_head' :[2.056], 'z_head' :[-2.65]})
#beam = beam_hit(pos_data,ans,0,-1,'x_head','y_head','z_head', 'x', 'y', 'z')

def beam_hit_mat(headset_position_data, left_gaze_mat, right_gaze_mat,):
    #This function returns you the coordinates that the subject looked at in Unity coordinate system 
    hit_mat_l = pd.DataFrame(columns=['x', 'y', 'z'])
    hit_mat_r = pd.DataFrame(columns=['x', 'y', 'z'])
    headset_position_data = headset_position_data.set_axis(['x_head','y_head','z_head'],axis=1, inplace=False)
    for i in range(len(headset_position_data)):
        print(i)
        right = right_gaze_mat.iloc[[i]]
        left = left_gaze_mat.iloc[[i]]
        head = headset_position_data.iloc[[i]]
        Rvec = beam_hit(head, right, 0, -1, 'x_head','y_head','z_head', 'x', 'y', 'z')
        Lvec = beam_hit(head, left, 0, -1, 'x_head','y_head','z_head', 'x', 'y', 'z')
        Rvec= np.reshape(Rvec,(1,3))
        Lvec = np.reshape(Lvec,(1,3))
        Rvec_dataFrame = pd.DataFrame(Rvec,columns=['x','y','z'])
        Lvec_dataFrame= pd.DataFrame(Lvec,columns=['x','y','z'])
        hit_mat_l = hit_mat_l.append(Lvec_dataFrame)
        hit_mat_r = hit_mat_r.append(Rvec_dataFrame)
    return hit_mat_l,hit_mat_r

def create_new_csv_output(old_csv_name,rightBeam,leftBeam):
    rightBeam = R_eye.fillna(0)
    leftBeam = L_eye.fillna(0)
    rightBeam.columns = ['right_x_beam', 'right_y_beam', 'right_z_beam']
    leftBeam.columns = ['left_x_beam', 'left_y_beam', 'left_z_beam']
    indexs = list(range(0,len(rightBeam)))
    rightBeam.index = indexs
    leftBeam.index= indexs
    old_data = pd.read_csv('TrackersOutputData.csv')
    all_data = pd.concat([old_data,leftBeam,rightBeam],axis=1)
    all_data.to_csv('NewTrackersOutputData.csv')
    return all_data
    ###exmple how to use the functions
a = load_data('TrackersOutputData.csv','HeadsetGlobalRotationEulerX','Y.5','Z.5','HeadSetGlobalPositionX','Y.6','Z.6','right.gaze_direction_normalized.x','right.gaze_direction_normalized.y','right.gaze_direction_normalized.z','left.gaze_direction_normalized.x','left.gaze_direction_normalized.y','left.gaze_direction_normalized.z')
right_gaze_mat,left_gaze_mat =from_mask_to_unity(a,('HeadsetGlobalRotationEulerX','Y.5','Z.5'),('right.gaze_direction_normalized.x','right.gaze_direction_normalized.y','right.gaze_direction_normalized.z'),('left.gaze_direction_normalized.x','left.gaze_direction_normalized.y','left.gaze_direction_normalized.z'))
right_gaze_mat.to_pickle("right_gaze_mat.pkl")
left_gaze_mat.to_pickle("left_gaze_mat.pkl")

head_position = a[['HeadSetGlobalPositionX','Y.6','Z.6']]
R_eye,L_eye = beam_hit_mat(head_position,left_gaze_mat,right_gaze_mat)
R_eye.to_pickle("R_eye.pkl")
L_eye.to_pickle("L_eye.pkl")
#R_eye = pd.read_pickle("R_eye.pkl")
#L_eye = pd.read_pickle("L_eye.pkl")
allData = create_new_csv_output('TrackersOutputData.csv',R_eye,L_eye)
#R_eye = pd.read_pickle("R_eye.pkl")
#L_eye = pd.read_pickle("L_eye.pkl")
R_eye.to_csv("right_eye_beam.csv")
L_eye.to_csv("lef_eye_beam.csv")


