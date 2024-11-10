import sim


def position_objectCoppelia(clientID):
    _, MTB_Robot=sim.simxGetObjectHandle(clientID, 'MTB_Robot', sim.simx_opmode_blocking)  # Robot completo
    _, MTB_axis1=sim.simxGetObjectHandle(clientID, 'MTB_axis1', sim.simx_opmode_blocking)  # Junta revolucion 1
    _, MTB_axis2=sim.simxGetObjectHandle(clientID, 'MTB_axis2', sim.simx_opmode_blocking)  # Junta revolucion 2
    _, MTB_axis3=sim.simxGetObjectHandle(clientID, 'MTB_axis3', sim.simx_opmode_blocking)  # Junta prismatica 3
    _, MTB_axis4=sim.simxGetObjectHandle(clientID, 'MTB_axis3', sim.simx_opmode_blocking)  # Junta revolucion 4
    
    _, suctionPadSensor=sim.simxGetObjectHandle(clientID, 'suctionPadSensor', sim.simx_opmode_blocking)  # Sensor de proximidad

    # _,position_MTB_Robot=sim.simxGetObjectPosition(clientID, MTB_Robot, -1, sim.simx_opmode_blocking)
    # _,orientation_MTB_axis1=sim.simxGetObjectOrientation(clientID, MTB_axis1, -1, sim.simx_opmode_blocking)
    # _,orientation_MTB_axis2=sim.simxGetObjectOrientation(clientID, MTB_axis2, -1, sim.simx_opmode_blocking)
    _,orientation_MTB_axis3=sim.simxGetObjectPosition(clientID, MTB_axis3, -1, sim.simx_opmode_blocking)
    # _,orientation_MTB_axis4=sim.simxGetObjectOrientation(clientID, MTB_axis4, -1, sim.simx_opmode_blocking)

    # res, dist, point, obj, n = sim.simxReadProximitySensor(clientID, suctionPadSensor, sim.simx_opmode_streaming)

    # print(res, dist, point, obj, n)


    # print(orientation_MTB_axis1)
    # print(orientation_MTB_axis2)
    # print(orientation_MTB_axis3)
    # print(orientation_MTB_axis4)
    orientation_MTB_axis3[-1]+=-0.01

    # sim.simxSetObjectPosition(clientID,MTB_Robot,-1,(0,1),sim.simx_opmode_oneshot)
    # _ = sim.simxSetObjectOrientation(clientID, MTB_axis1, -1 , (0, 0, 0.5), sim.simx_opmode_oneshot)  # Angulos de Euler (alpha, beta, gamma)
    # _ = sim.simxSetObjectOrientation(clientID, MTB_axis2, -1 , (0, 0, 0.5), sim.simx_opmode_oneshot)  # Angulos de Euler (alpha, beta, gamma)
    _ = sim.simxSetObjectPosition(clientID, MTB_axis3, -1, tuple(orientation_MTB_axis3), sim.simx_opmode_oneshot)  # Distancias en (x, y, z)
    # _ = sim.simxSetObjectOrientation(clientID, MTB_axis4, -1 , (0, 0, 0.5), sim.simx_opmode_oneshot)  # Angulos de Euler (alpha, beta, gamma)

    _, orientation_MTB_Robot=sim.simxGetObjectOrientation(clientID, MTB_Robot, -1, sim.simx_opmode_blocking)
    # _, orientation_MTB_axis2=sim.simxGetObjectOrientation(clientID, MTB_axis2, -1, sim.simx_opmode_blocking)
    # _, orientation_MTB_axis3=sim.simxGetObjectOrientation(clientID, MTB_axis3, -1, sim.simx_opmode_blocking)
    # _, orientation_MTB_axis4=sim.simxGetObjectOrientation(clientID, MTB_axis4, -1, sim.simx_opmode_blocking)

    # print(orientation_MTB_axis1)
    # print(orientation_MTB_axis2)
    # print(orientation_MTB_axis3)
    # print(orientation_MTB_axis4)

    # res, dist, point, obj, n = sim.simxhandle handleProximitySensor(int sensorHandle)


    returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(clientID, suctionPadSensor, sim.simx_opmode_streaming)

    print(returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector)


    # _,position_MTB_Robot=sim.simxGetObjectPosition(clientID, MTB_Robot, -1, sim.simx_opmode_blocking)
    # _,orientation_MTB_Robot=sim.simxGetObjectOrientation(clientID, MTB_Robot, -1, sim.simx_opmode_blocking)

    # print(position_MTB_Robot)

# def move_Coppelia(w_WheelLeft,w_WheelRight):
#     global MTB, clientID
#     ret,wheelR=sim.simxGetObjectHandle(clientID,'MotorR',sim.simx_opmode_blocking)
#     ret,wheelL=sim.simxGetObjectHandle(clientID,'MotorL',sim.simx_opmode_blocking)
#     q1 = w_WheelRight/6.5
#     _ = sim.simxSetJointTargetVelocity(clientID, wheelR, q1, sim.simx_opmode_blocking)
#     q2 = w_WheelLeft/6.5
#     _ = sim.simxSetJointTargetVelocity(clientID, wheelL, q2, sim.simx_opmode_blocking) 
#     _,position_Mobil=sim.simxGetObjectPosition(clientID, MTB, -1, sim.simx_opmode_blocking)
#     _,orientation_Mobil=sim.simxGetObjectOrientation(clientID, MTB, -1, sim.simx_opmode_blocking)
#     return position_Mobil,orientation_Mobil


port=19999
sim.simxFinish(-1) 
clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5)
if clientID == 0: print("conectado a", port)
else: print("no se pudo conectar")


position_objectCoppelia(clientID)