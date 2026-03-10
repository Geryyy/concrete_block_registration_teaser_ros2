#pragma once

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include "pcd_block_estimation/template_utils.hpp"

namespace concrete_block_registration_teaser
{

struct PreprocessingParams
{
  size_t max_pts{500};
  int nb_neighbors{20};
  double std_dev{2.0};
  bool enable_cluster_filter{false};
  double cluster_eps{0.08};
  int cluster_min_points{20};
  int cluster_min_size{100};
};

struct GlobalRegistrationParams
{
  static constexpr int MAX_PLANES = 2;
  Eigen::Vector3d Z_WORLD{0, 0, 1};
  double dist_thresh{0.02};
  int min_inliers{100};
  double angle_thresh{0.9};
  double max_plane_center_dist{0.6};
  bool enable_plane_clipping{false};
  bool reject_tall_vertical{true};
};

struct LocalRegistrationParams
{
  double icp_dist{0.04};
  bool relax_num_faces_match{false};
  bool use_fk_translation_seed{false};
  std::vector<double> icp_dist_multipliers{1.0, 1.5, 2.0};
  bool enable_point_to_point_fallback{true};
};

struct TeaserRegistrationParams
{
  double noise_bound{0.02};
  double cbar2{1.0};
  bool estimate_scaling{false};
  double rotation_gnc_factor{1.4};
  int rotation_max_iterations{100};
  double rotation_cost_threshold{1e-6};
  double max_clique_time_limit_s{0.2};
  size_t min_correspondences{30};
  size_t max_template_points{1000};
  double nn_corr_max_dist{0.08};
  bool enable_icp_refinement{true};
  double icp_refine_dist{0.04};
  double eval_corr_dist{0.04};
};

struct RegistrationInput
{
  open3d::geometry::PointCloud scene;
  cv::Mat mask;
  Eigen::Matrix4d T_world_cloud;
  bool has_translation_seed_world{false};
  Eigen::Vector3d translation_seed_world{Eigen::Vector3d::Zero()};
};

struct RegistrationOutput
{
  bool success{false};
  Eigen::Matrix4d T_world_block{Eigen::Matrix4d::Identity()};
  double fitness{0.0};
  double rmse{0.0};
  int template_index{-1};
  std::string failure_stage;
  std::string failure_reason;
  open3d::geometry::PointCloud debug_scene;
};

struct BlockRegistrationConfig
{
  std::string world_frame;

  Eigen::Matrix4d T_P_C;
  Eigen::Matrix3d K;
  std::vector<pcd_block::TemplateData> templates;

  PreprocessingParams preproc;
  GlobalRegistrationParams glob;
  LocalRegistrationParams local;
  TeaserRegistrationParams teaser;

  bool publish_debug_cutout{true};
  bool publish_debug_mask{true};
  bool verbose_logs{true};
  bool dump_enabled{false};
  bool dump_failure_package{true};
  std::string dump_dir;
  std::string fk_seed_tcp_frame{"elastic/K8_tool_center_point"};
  Eigen::Vector3d fk_seed_tcp_to_block_xyz{Eigen::Vector3d::Zero()};
};

}  // namespace concrete_block_registration_teaser
