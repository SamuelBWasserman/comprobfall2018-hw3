import rospy
from gazebo_msgs.msg import ModelState
from turtlebot_ctrl.srv import TurtleBotControl
from turtlebot_ctrl.msg import TurtleBotScan
from std_msgs.msg import Float32
from std_msgs.msg import Bool

def move_robot(heading, distance, return_ground_truth):
    rospy.wait_for_service('turtlebot_ctrl')
    turtlebot_ctrl = rospy.ServiceProxy('turtlebot_ctrl', TurtleBotControl)
    try:
        resp = turtlebot_ctrl(heading, distance, return_ground_truth)
        print str(resp)
        return resp
    except rospy.ServiceException as exc:
        print "Service did not process request: " + str(exc)
