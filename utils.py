import numpy as np


def getPoints(outLiers):
    movingPoints = np.array([
                            [345.7500,  416.7500]
                            ,[106.2500,  334.2500]
                            ,[414.2500,  163.7500]
                            ,[256.7500,  458.2500]
                            ,[366.7500,  320.2500]
                            ,[263.2500,  306.2500]
                            ,[198.7500,  122.7500]
                            ,[372.7500,  187.7500]
                            ,[425.7500,  369.7500]])
    fixedPoints = np.array([
                            [247.7500,  440.2500]
                            ,[88.2500,  243.2500]
                            ,[439.7500,  256.2500]
                            ,[150.2500,  426.7500]
                            ,[316.2500,  370.2500]
                            ,[236.2500,  303.2500]
                            ,[279.2500,  112.7500]
                            ,[392.7500,  261.2500]
                            ,[341.7500,  442.2500]])
    if (outLiers == 'no_outliers'):
        pass
    elif (outLiers == 'with_outliers'):
        movingOutLiers = np.array([[211.7500,  393.2500], [119.2500,  164.2500],[298.2500,  110.7500]])
        movingPoints= np.append(movingPoints, movingOutLiers,axis=0)
        fixedOutLiers = np.array([[119.7500,  271.7500],[158.7500,  137.2500],[361.7500,  145.7500]])
        fixedPoints = np.append(fixedPoints, fixedOutLiers,axis=0)
    else:
        print("Unknown string " + outLiers)
        exit()
    return fixedPoints, movingPoints


def ransac(x,y,funcFindF,funcDist,minPtNum,iterNum,thDist,thInlrRatio):
    """
    Use RANdom SAmple Consensus to find a fit from X to Y.
    :param x: M*n matrix including n points with dim M
    :param y: N*n matrix including n points with dim N
    :param funcFindF: a function with interface f1 = funcFindF(x1,y1) where:
                x1 is M*n1
                y1 is N*n1 n1 >= minPtNum
                f is an estimated transformation between x1 and y1 - can be of any type
    :param funcDist: a function with interface d = funcDist(x1,y1,f) that uses f returned by funcFindF and returns the
                distance between <x1 transformed by f> and <y1>. d is 1*n1.
                For line fitting, it should calculate the distance between the line and the points [x1;y1];
                For homography, it should project x1 to y2 then calculate the dist between y1 and y2.
    :param minPtNum: the minimum number of points with whom can we find a fit. For line fitting, it's 2. For
                homography, it's 4.
    :param iterNum: number of iterations (== number of times we draw a random sample from the points
    :param thDist: inlier distance threshold.
    :param thInlrRatio: ROUND(THINLRRATIO*n) is the inlier number threshold
    :return: [f, inlierIdx] where: f is the fit and inlierIdx are the indices of inliers

    transalated from matlab by Adi Szeskin.
    """

    ptNum = len(x)
    thInlr = round(thInlrRatio*ptNum)


    inlrNum = np.zeros([iterNum,1])
    fLib= np.zeros(shape=(iterNum,3,3))
    for i in range(iterNum):
        permut = np.random.permutation(ptNum)
        sampleIdx = permut[range(minPtNum)]
        f1 = funcFindF(x[sampleIdx,:],y[sampleIdx,:])
        dist = funcDist(x,y,f1)
        b = dist<=thDist
        r = np.array(range(len(b)))
        inlier1 = r[b]
        inlrNum[i] = len(inlier1)
        if len(inlier1) < thInlr: continue
        fLib[i] = funcFindF(x[inlier1,:],y[inlier1,:])

    idx = inlrNum.tolist().index(max(inlrNum))
    f = fLib[idx]
    dist = funcDist(x,y,f);
    b = dist<=thDist
    r = np.array(range(len(b)))
    inlierIdx = r[b]
    return f, inlierIdx

