// Copyright (C) 2017  Juha Reunanen (juha.reunanen@tomaattinen.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_FIND_OPTIMAL_THRESHOLD_H_
#define DLIB_DNn_FIND_OPTIMAL_THRESHOLD_H_

// TODO: add documentation
//#include "find_optimal_threshold_abstract.h"

#include "../image_processing/box_overlap_testing.h"
#include "../image_processing/full_object_detection.h"

#include <unordered_set>
#include <unordered_map>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    double find_optimal_threshold(const std::vector<roc_point>& roc_curve, double max_margin_if_single_optimal_point)
    {
        DLIB_CASSERT(!roc_curve.empty());

        // see https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        const auto youden_index = [](const roc_point& roc_point)
        {
            return roc_point.true_positive_rate - roc_point.false_positive_rate;
        };

        const auto first_point = std::max_element(roc_curve.begin(), roc_curve.end(),
            [youden_index](const roc_point& lhs, const roc_point& rhs) {
                return youden_index(lhs) < youden_index(rhs);
            });

        const auto first_point_youden_index = youden_index(*first_point);
        auto last_point = first_point;

        const auto next_point_exists_and_is_as_good_as_first_point = [&]()
        {
            const auto next_point = std::next(last_point);
            return next_point != roc_curve.end()
                && youden_index(*next_point) == first_point_youden_index;
        };

        while (next_point_exists_and_is_as_good_as_first_point()) {
            ++last_point;
        }

        DLIB_CASSERT(youden_index(*first_point) == youden_index(*last_point));

        if (first_point->detection_threshold == last_point->detection_threshold)
        {
            // Single optimal point: subtract a margin.
            const auto next_point = std::next(last_point);
            if (next_point == roc_curve.end())
            {
                return first_point->detection_threshold - max_margin_if_single_optimal_point;
            }
            DLIB_CASSERT(youden_index(*first_point) > youden_index(*next_point));
            DLIB_CASSERT(first_point->detection_threshold >= next_point->detection_threshold);
            const auto distance = first_point->detection_threshold - next_point->detection_threshold;
            const auto margin = std::min(max_margin_if_single_optimal_point, distance / 2.0);
            return first_point->detection_threshold - margin;
        }
        else
        {
            // More than one optimal point: take the middle ground.
            return (first_point->detection_threshold + last_point->detection_threshold) / 2.0;
        }
    }

    double find_optimal_threshold(
        const std::vector<dlib::loss_mmod_::mmod_training_input>& truth,
        const std::vector<std::vector<mmod_rect>>& detections,
        double truth_match_iou_threshold_for_correct_label = 0.25,
        double truth_match_iou_threshold_for_incorrect_label = 0.5,
        double truth_match_iou_threshold_for_ignore = 0.25
    )
    {
        DLIB_CASSERT(!detections.empty());
        DLIB_CASSERT(truth.size() == detections.size());

        std::vector<double> true_detections, false_detections;

        double minimum_detection_confidence = std::numeric_limits<double>::max();
        double maximum_detection_confidence = -std::numeric_limits<double>::max();

        const int sample_count = static_cast<int>(truth.size());

#pragma omp parallel for
        for (int i = 0; i < sample_count; ++i)
        {
            const auto& truth_i = truth[i];
            const std::vector<mmod_rect>& detections_i = detections[i];

            const size_t detections_i_size = detections_i.size();
            const size_t truth_i_size = truth_i.mmod_rects.size();

            //                 truth index       confidence, detection index
            std::unordered_map<size_t, std::pair<double, size_t>> best_matching_truths;
            std::unordered_set<size_t> detection_indexes_that_can_be_ignored;
            std::unordered_set<size_t> detection_indexes_that_should_be_ignored;

            for (size_t j = 0; j < detections_i_size; ++j)
            {
                const mmod_rect& detection = detections_i[j];
                const auto detection_center = center(detection.rect);

                if (dlib::loss_mmod_::should_point_be_ignored(detection_center, truth_i.mask))
                {
                    detection_indexes_that_should_be_ignored.insert(j);
                    continue;
                }

                bool found_corresponding_ignore_truth = false;

                for (size_t k = 0; k < truth_i_size; ++k)
                {
                    const mmod_rect& candidate_truth = truth_i.mmod_rects[k];
                    const double truth_match_iou = box_intersection_over_union(detection.rect, candidate_truth.rect);

                    const auto accept_truth_hit = [&]()
                    {
                        if (candidate_truth.ignore)
                        {
                            return false;
                        }
                        if (dlib::loss_mmod_::should_point_be_ignored(center(candidate_truth.rect), truth_i.mask))
                        {
                            return false;
                        }
                        if (truth_match_iou >= truth_match_iou_threshold_for_correct_label && detection.label == candidate_truth.label)
                        {
                            return true; // accept with correct label
                        }
                        if (truth_match_iou >= truth_match_iou_threshold_for_incorrect_label)
                        {
                            return true; // accept with incorrect label
                        }
                        return false;
                    };

                    const auto accept_ignore = [&]()
                    {
                        if (candidate_truth.ignore && truth_match_iou >= truth_match_iou_threshold_for_ignore)
                        {
                            return true;
                        }
                        if (dlib::loss_mmod_::should_point_be_ignored(center(candidate_truth.rect), truth_i.mask))
                        {
                            return true;
                        }
                        return false;
                    };

                    if (accept_truth_hit())
                    {
                        const auto confidence_and_detection_index = [&]() { return std::make_pair(detection.detection_confidence, j); };
                        const auto i = best_matching_truths.find(k);
                        if (i == best_matching_truths.end())
                        {
                            best_matching_truths[k] = confidence_and_detection_index();
                        }
                        else if (detection.detection_confidence > i->second.first)
                        {
                            i->second = confidence_and_detection_index();
                        }
                    }

                    if (accept_ignore())
                    {
                        found_corresponding_ignore_truth = true;
                    }
                }

                if (found_corresponding_ignore_truth)
                {
                    detection_indexes_that_can_be_ignored.insert(j);
                }
            }

            std::unordered_set<size_t> detection_indexes_that_have_corresponding_best_match_truth;

            for (const auto& j : best_matching_truths)
            {
                detection_indexes_that_have_corresponding_best_match_truth.insert(j.second.second);
            }

#pragma omp critical
            for (size_t j = 0; j < detections_i_size; ++j)
            {
                const mmod_rect& detection = detections_i[j];

                const auto best_matching_truth_found = [&]()
                {
                    return detection_indexes_that_have_corresponding_best_match_truth.find(j) != detection_indexes_that_have_corresponding_best_match_truth.end();
                };

                const auto should_be_ignored = [&]()
                {
                    return detection_indexes_that_should_be_ignored.find(j) != detection_indexes_that_should_be_ignored.end();
                };

                const auto can_be_ignored = [&]()
                {
                    return detection_indexes_that_can_be_ignored.find(j) != detection_indexes_that_can_be_ignored.end();
                };

                if (!should_be_ignored())
                {
                    if (best_matching_truth_found())
                    {
                        true_detections.push_back(detection.detection_confidence);
                    }
                    else if (!can_be_ignored())
                    {
                        false_detections.push_back(detection.detection_confidence);
                    }
                }

                minimum_detection_confidence = std::min(minimum_detection_confidence, detection.detection_confidence);
                maximum_detection_confidence = std::max(maximum_detection_confidence, detection.detection_confidence);
            }
        }

        if (true_detections.empty() && false_detections.empty())
        {
            return 0.0; // TODO: what should we do here?
        }

        constexpr double epsilon = 1e-6;
        if (true_detections.empty())
        {
            return maximum_detection_confidence + epsilon;
        }
        else if (false_detections.empty())
        {
            return minimum_detection_confidence - epsilon;
        }

        const auto roc_curve = compute_roc_curve(true_detections, false_detections);

        return find_optimal_threshold(roc_curve, epsilon);
    }
// ----------------------------------------------------------------------------------------


}

#endif // DLIB_DNn_FIND_OPTIMAL_THRESHOLD_H_

