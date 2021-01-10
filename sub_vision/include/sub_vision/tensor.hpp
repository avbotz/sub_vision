/*
 * MIT License
 *
 * Copyright (c) 2019 Sergio Izquierdo
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/** @file tensor.hpp
 *  @brief TF Object Detection API C++ wrapper for tensors.
 *
 *  @author Sergio Izquierdo
 */
#ifndef CPPFLOW_TENSOR_H
#define CPPFLOW_TENSOR_H

#include <tensorflow/c/c_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include "sub_vision/model.hpp"

class Model;

class Tensor {
public:
    Tensor(const Model& model, const std::string& operation);
    ~Tensor();

    void clean();

    template<typename T> void set_data(std::vector<T> new_data);
    template<typename T> void set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape);
    template<typename T> std::vector<T> get_data();


private:
    TF_Tensor* val;
    TF_Output op;
    TF_DataType type;
    std::vector<int64_t> shape;
    std::vector<int64_t>* actual_shape;
    void* data;
    int flag;

    // Aux functions
    void error_check(bool condition, const std::string& error);
    template <typename T> static TF_DataType deduce_type();
    void deduce_shape(const Model& model);

public:
    friend class Model;
};

#endif //CPPFLOW_TENSOR_H
