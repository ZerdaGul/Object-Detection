#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from turtlesim.srv import Kill
from turtlesim.srv import Spawn
import random
from std_msgs.msg import Int16

class TurtleCatcherNode(Node):
	def __init__(self):
		
		super().__init__("turtlec")
		self.catch_number=Int16()
		
		self.turtle1_pose = Pose()
		self.turtle2_pose = Pose()
		self.catch_number.data = 0
		self.spawn()



		self.publisher_=self.create_publisher(Int16 ,"catch_number", 10)
		self.subscriber_=self.create_subscription(Int16 ,"catch_number", self.coughted,10)
		
		self.subscriber1= self.create_subscription(Pose, "/turtle1/pose", self.save1, 10)
		
		

		
		self.get_logger().info("Node has been started")

	def coughted(self,msg):

		self.catch_number = msg
		#self.get_logger().info(self.catch_number)

	def publish_number(self):
		msg= Int16()
		msg= self.catch_number

		self.publisher_.publish(msg)

	def save1(self,msg):
		print(f"/turtle{self.catch_number.data+5}/pose")
		self.turtle1_pose = msg
		print("t", msg)
		self.subscriber2= self.create_subscription(Pose, f"/turtle{self.catch_number.data+5}/pose", self.save2, 10)

	def save2(self, msg):

		self.turtle2_pose= msg
		print("a",msg)
		self.compare()
		self.publish_number()
		
	def compare(self):
		
		tolerantx= float(self.turtle1_pose.x - self.turtle2_pose.x)
		toleranty= float(self.turtle1_pose.y - self.turtle2_pose.y)
		print(tolerantx," ",toleranty)
		if tolerantx<0.1 and toleranty< 0.1:
			self.kill()
			self.spawn()

	def spawn(self):
		self.cli_spawn= self.create_client(Spawn, "spawn")
		req = Spawn.Request()
		req.x = random.uniform(2,10)
		req.y = random.uniform(2,10)

		req.name = "turtle" + str(self.catch_number.data+5)
		self.cli_spawn.call_async(req)

	def kill(self):
		self.cli_kill = self.create_client(Kill, 'kill')
		req= Kill.Request()
		req.name = f"turtle{self.catch_number.data+5}"
		self.catch_number.data += 1
		self.cli_kill.call_async(req)

def main(args=None):
	rclpy.init(args=args)
	node = TurtleCatcherNode() 
	
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__":
	main()
