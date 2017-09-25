// Copyright (C) 2017  Juha Reunanen (juha.reunanen@tomaattinen.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_FIND_OPTIMAL_THRESHOLD_H_
#define DLIB_DNn_FIND_OPTIMAL_THRESHOLD_H_

// TODO: add documentation
//#include "find_optimal_threshold_abstract.h"

#include "../image_processing/box_overlap_testing.h"
#include "../image_processing/full_object_detection.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    double find_optimal_threshold(const std::vector<roc_point>& roc_curve)
    {
        DLIB_CASSERT(!roc_curve.empty());

        // see https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        const auto youden_index = [](const roc_point& roc_point)
        {
            return roc_point.true_positive_rate - roc_point.false_positive_rate;
        };

        const auto point1 = std::max_element(roc_curve.begin(), roc_curve.end(),
            [youden_index](const roc_point& lhs, const roc_point& rhs) {
                return youden_index(lhs) < youden_index(rhs);
            });

        const auto point2 = point1 + 1;
        if (point2 == roc_curve.end()) {
            return point1->detection_threshold;
        }
        DLIB_CASSERT(point1->detection_threshold >= point2->detection_threshold);
        return (point1->detection_threshold + point2->detection_threshold) / 2.0;
    }

    double find_optimal_threshold(
        const std::vector<std::vector<mmod_rect>>& truth,
        const std::vector<std::vector<mmod_rect>>& detections,
        double truth_match_iou_threshold_for_correct_label = 0.1,
        double truth_match_iou_threshold_for_incorrect_label = 0.5
    )
    {
        DLIB_CASSERT(!detections.empty());
        DLIB_CASSERT(truth.size() == detections.size());

        std::vector<double> true_detections, false_detections;

        double minimum_detection_confidence = std::numeric_limits<double>::max();
        double maximum_detection_confidence = -std::numeric_limits<double>::max();

        for (size_t i = 0, sample_count = truth.size(); i < sample_count; ++i)
        {
            const std::vector<mmod_rect>& truth_i = truth[i];
            const std::vector<mmod_rect>& detections_i = detections[i];

            const int detections_i_size = static_cast<int>(detections_i.size());

#pragma omp parallel for
            for (int j = 0; j < detections_i_size; ++j)
            {
                const mmod_rect& detection = detections_i[j];

                bool found_corresponding_truth = false;
                bool found_corresponding_ignore = false;
                for (const mmod_rect& candidate_truth : truth_i)
                {
                    const double truth_match_iou = box_intersection_over_union(detection.rect, candidate_truth.rect);
                    const auto accept_with_correct_label = [&]()
                    {
                        return truth_match_iou >= truth_match_iou_threshold_for_correct_label && detection.label == candidate_truth.label;
                    };
                    if (accept_with_correct_label() || truth_match_iou >= truth_match_iou_threshold_for_incorrect_label)
                    {
                        if (candidate_truth.ignore)
                        {
                            found_corresponding_ignore = true;
                        }
                        else
                        {
                            found_corresponding_truth = true;
                            break;
                        }
                    }
                }

#pragma omp critical
                {
                    if (found_corresponding_truth)
                    {
                        true_detections.push_back(detection.detection_confidence);
                    }
                    else if (!found_corresponding_ignore)
                    {
                        false_detections.push_back(detection.detection_confidence);
                    }

                    minimum_detection_confidence = std::min(minimum_detection_confidence, detection.detection_confidence);
                    maximum_detection_confidence = std::max(maximum_detection_confidence, detection.detection_confidence);
                }
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

        return find_optimal_threshold(roc_curve);
    }
// ----------------------------------------------------------------------------------------


}

#endif // DLIB_DNn_FIND_OPTIMAL_THRESHOLD_H_

