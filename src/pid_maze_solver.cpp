#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <vector>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm> // Include this for std::clamp

using namespace std::chrono_literals;

class MazeSolver : public rclcpp::Node
{
public:
    MazeSolver(int scene_number) 
        : Node("distance_controller"), current_goal_index_(0), scene_number_(scene_number),
          max_linear_speed_(0.8), max_angular_speed_(0.8)  // Set maximum speeds here
    {
        // Initialize publisher
        velocity_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        // Initialize subscriber
        odometry_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10, std::bind(&MazeSolver::odometryCallback, this, std::placeholders::_1));

        // Hardcoded distance goals in meters based on the scene number
        if (scene_number_ == 1) {
            // Simulation waypoints
            distance_goals_ = {
                {0.460, -0.050},
                {0.460, -1.367},
                {1.026, -1.367},
                {1.026, -0.877},
                {1.387, -0.877},
                {1.387, -0.357},
                {1.933, -0.357},
                {1.933, 0.568},
                {1.406, 0.568},
                {1.406, 0.191},
                {0.918, 0.191},
                {0.650, 0.522},
                {0.102, 0.522}
            };
        } else if (scene_number_ == 2) {
            // CyberWorld waypoints (adjusted for starting position)
            distance_goals_ = {
                {1.731, 0.508},
                {1.910, 0.164},
                {0.931, -0.478},
                {1.205, -0.916},
                {1.636, -0.688},
                {1.771, -1.263},
                {1.416, -1.351},
                {1.389, -1.796},
                {0.876, -1.680},
                {0.932, -0.899},
                {0.454, -0.802},
                {0.592, -0.361},
                {0.184, -0.237}
            };
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid scene number: %d", scene_number_);
            rclcpp::shutdown();
            return;
        }

        // PID parameters
        this->declare_parameter<double>("kp_l", 1.5);
        this->declare_parameter<double>("ki_l", 0.001);
        this->declare_parameter<double>("kd_l", 1.0);

        this->declare_parameter<double>("kp_r", 2.0);
        this->declare_parameter<double>("ki_r", 0.001);
        this->declare_parameter<double>("kd_r", 0.5);

        // Use the declared parameters
        kp_l_ = this->get_parameter("kp_l").as_double();
        ki_l_ = this->get_parameter("ki_l").as_double();
        kd_l_ = this->get_parameter("kd_l").as_double();

        kp_r_ = this->get_parameter("kp_r").as_double();
        ki_r_ = this->get_parameter("ki_r").as_double();
        kd_r_ = this->get_parameter("kd_r").as_double();

        // Initialize PID control variables
        integral_l_ = 0.0;
        previous_error_l_ = 0.0;
        integral_r_ = 0.0;
        previous_error_r_ = 0.0;
        control_traslation_ = false;
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;
    std::vector<std::vector<double>> distance_goals_;
    int current_goal_index_;
    double kp_l_, ki_l_, kd_l_;
    double kp_r_, ki_r_, kd_r_;
    double integral_l_, previous_error_l_;
    double integral_r_, previous_error_r_;
    bool control_traslation_;
    int scene_number_;
    double max_linear_speed_; // Maximum linear speed
    double max_angular_speed_; // Maximum angular speed

    Eigen::Vector2d transformCoordinates(const Eigen::Vector2d& p_CA, const Eigen::Vector2d& p_BA, double theta) {
        // Create the rotation matrix using Eigen
        Eigen::Matrix2d R_BA;
        R_BA << cos(theta), -sin(theta),
                sin(theta), cos(theta);

        // Calculate the new position of C relative to B
        Eigen::Vector2d p_CB = R_BA.transpose() * (p_CA - p_BA);
        RCLCPP_DEBUG(this->get_logger(), "[%d] X: %.3f\tY: %.3f", current_goal_index_, p_CB[0], p_CB[1]);
        return p_CB;
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double z = msg->pose.pose.orientation.z;
        double w = msg->pose.pose.orientation.w;
        double current_x = msg->pose.pose.position.x;
        double current_y = msg->pose.pose.position.y;
        double current_rotation = 2 * std::atan2(z, w);
        auto cmd_vel = geometry_msgs::msg::Twist();
        RCLCPP_DEBUG(this->get_logger(), "[%d] Current X: %.3f", current_goal_index_, current_x);
        RCLCPP_DEBUG(this->get_logger(), "[%d] Current Y: %.3f", current_goal_index_, current_y);
        RCLCPP_DEBUG(this->get_logger(), "[%d] Current rotation: %.3f", current_goal_index_, current_rotation * 180 / M_PI);

        if (current_goal_index_ < int(distance_goals_.size())) {
            std::vector<double> waypoint = distance_goals_[current_goal_index_];
            Eigen::Vector2d p_BA(current_x, current_y);
            Eigen::Vector2d p_CA(waypoint[0], waypoint[1]);
            Eigen::Vector2d p_CB = transformCoordinates(p_CA, p_BA, current_rotation);
            double error = std::atan2(p_CB[1], p_CB[0]);
            RCLCPP_DEBUG(this->get_logger(), "[%d] Angular error: %.3f", current_goal_index_, error);
            double derivative = error - previous_error_r_;
            integral_r_ += error;
            previous_error_r_ = error;

            double control_signal = kp_r_ * error + ki_r_ * integral_r_ + kd_r_ * derivative;
            control_signal = std::clamp(control_signal, -max_angular_speed_, max_angular_speed_); // Clamp to max angular speed

            // Publish the velocity command
            cmd_vel.angular.z = control_signal;
            velocity_publisher_->publish(cmd_vel);

            // Check if goal is reached within a small threshold
            if (std::abs(error) < 0.01 && !control_traslation_)
            {
                RCLCPP_INFO(this->get_logger(), "[%d] Finished rotation", current_goal_index_);
                integral_r_ = 0;  // Reset integral term
                control_traslation_ = true;
            }
        }
        else {
            // Stop the robot if all goals are reached
            auto stop_cmd = geometry_msgs::msg::Twist();
            stop_cmd.angular.z = 0;
            stop_cmd.linear.x = 0;
            velocity_publisher_->publish(stop_cmd);
            rclcpp::shutdown();
            return;
        }

        if (control_traslation_) {
            double goal_x = distance_goals_[current_goal_index_][0];
            double goal_y = distance_goals_[current_goal_index_][1];
            double error = std::sqrt(std::pow(current_x - goal_x, 2) + std::pow(current_y - goal_y, 2));
            RCLCPP_DEBUG(this->get_logger(), "[%d] Linear error: %.3f", current_goal_index_, error);
            double derivative = error - previous_error_l_;
            integral_l_ += error;
            previous_error_l_ = error;

            double control_signal = kp_l_ * error + ki_l_ * integral_l_ + kd_l_ * derivative;
            control_signal = std::clamp(control_signal, -max_linear_speed_, max_linear_speed_); // Clamp to max linear speed

            // Publish the velocity command
            cmd_vel.linear.x = control_signal;
            velocity_publisher_->publish(cmd_vel);

            // Check if goal is reached within a small threshold
            if (std::abs(error) < 0.01){
                RCLCPP_INFO(this->get_logger(), "Finished traslation");
                current_goal_index_++;
                integral_l_ = 0;  // Reset integral term
                control_traslation_ = false;
            }
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    int scene_number = 1; // Default to simulation
    if (argc > 1) {
        scene_number = std::atoi(argv[1]);
    }

    auto node = std::make_shared<MazeSolver>(scene_number);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}