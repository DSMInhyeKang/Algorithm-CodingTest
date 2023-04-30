//
//  main.swift
//  ProgrammersCodingTest
//
//  Created by 강인혜 on 2023/04/30.
//

import Foundation

// 중복된 문자 제거(120888)

func solution(_ my_string: String) -> String {
    var result = ""
    for str in my_string where !result.contains(str) {
        result.append(str)
    }
    return result
}
