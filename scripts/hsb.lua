-- Wireshark dissector for HSB ECB packets.

hsb_proto = Proto("hsb", "Holoscan Sensor Bridge")

local OPCODE = {
    [0x04]="WR_DWORD",
    [0x09]="WR_BLOCK",
    [0x84]="WR_DWORD reply",
    [0x89]="WR_BLOCK reply",
    [0x14]="RD_DWORD",
    [0x94]="RD_DWORD reply",
    [0x19]="RD_BLOCK",
    [0x99]="RD_BLOCK reply",
}

local ADDRESS = {
    [0x00000080]="FPGA_VERSION",
    [0x00000084]="FPGA_DATE",
}

hsb_opcode = ProtoField.uint8("hsb.opcode", "Opcode", base.HEX, OPCODE)
hsb_flags = ProtoField.uint8("hsb.flags", "Flags", base.HEX)
hsb_sequence = ProtoField.uint16("hsb.sequence", "Seq")
hsb_address = ProtoField.uint32("hsb.address", "Address", base.HEX, ADDRESS)
hsb_value = ProtoField.uint32("hsb.value", "Value", base.HEX)
hsb_response_code = ProtoField.uint8("hsb.response", "Response", base.HEX)
hsb_reserved = ProtoField.uint8("hsb.reserved", "Reserved", base.HEX)

hsb_proto.fields = {
    hsb_opcode,
    hsb_flags,
    hsb_sequence,
    hsb_address,
    hsb_value,
    hsb_response_code,
    hsb_reserved,
}

function hsb_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "HSB"
    local subtree = tree:add(hsb_proto, buffer(), "Holoscan sensor bridge")
    local offset = 0
    local t_opcode = subtree:add(hsb_opcode, buffer(offset, 1))
    local opcode = buffer(offset, 1):uint()
    offset = offset + 1
    local t_flags = subtree:add(hsb_flags, buffer(offset, 1))
    offset = offset + 1
    local t_sequence = subtree:add(hsb_sequence, buffer(offset, 2))
    offset = offset + 2

    if opcode == 0x04  -- WR_DWORD
    then
        local t_reserved = subtree:add(hsb_reserved, buffer(offset, 2))
        offset = offset + 2
        local t_address = subtree:add(hsb_address, buffer(offset, 4))
        offset = offset + 4
        local t_value = subtree:add(hsb_value, buffer(offset, 4))
        offset = offset + 4
    elseif opcode == 0x09  -- WR_BLOCK
    then
        local t_reserved = subtree:add(hsb_reserved, buffer(offset, 2))
        offset = offset + 2
        while offset + 8 <= buffer:len() do
            local t_address = subtree:add(hsb_address, buffer(offset, 4))
            offset = offset + 4
            local t_value = subtree:add(hsb_value, buffer(offset, 4))
            offset = offset + 4
        end
    elseif opcode == 0x84  -- WR_DWORD reply
    then
        local t_response = subtree:add(hsb_response_code, buffer(offset, 1))
        offset = offset + 1
        local t_reserved = subtree:add(hsb_reserved, buffer(offset, 1))
        offset = offset + 1
        local t_address = subtree:add(hsb_address, buffer(offset, 4))
        offset = offset + 4
    elseif opcode == 0x89  -- WR_BLOCK reply
    then
        local t_response = subtree:add(hsb_response_code, buffer(offset, 1))
        offset = offset + 1
        local t_reserved = subtree:add(hsb_reserved, buffer(offset, 1))
        offset = offset + 1
    elseif opcode == 0x14  -- RD_DWORD
    then
        local t_reserved = subtree:add(hsb_reserved, buffer(offset, 2))
        offset = offset + 2
        local t_address = subtree:add(hsb_address, buffer(offset, 4))
        offset = offset + 4
    elseif opcode == 0x94  -- RD_DWORD reply
    then
        local t_response = subtree:add(hsb_response_code, buffer(offset, 1))
        offset = offset + 1
        local t_reserved = subtree:add(hsb_reserved, buffer(offset, 1))
        offset = offset + 1
        local t_address = subtree:add(hsb_address, buffer(offset, 4))
        offset = offset + 4
        local t_value = subtree:add(hsb_value, buffer(offset, 4))
        offset = offset + 4
    end
end

--

hsb_enumeration_proto = Proto("hsb-enum", "Holoscan Sensor Bridge Enumeration")

hsb_enumeration_board_id = ProtoField.uint8("hsb-enumeration.board_id", "ID", base.HEX)
hsb_enumeration_board_version = ProtoField.bytes("hsb-enumeration.board_version", "Version", base.SPACE)
hsb_enumeration_serial_number = ProtoField.bytes("hsb-enumeration.serial_number", "Serial number", base.SPACE)
hsb_enumeration_cpnx_version = ProtoField.uint16("hsb-enumeration.cpnx_version", "CPNX version", base.HEX)
hsb_enumeration_cpnx_crc = ProtoField.uint16("hsb-enumeration.cpnx_crc", "CPNX CRC", base.HEX)
hsb_enumeration_clnx_version = ProtoField.uint16("hsb-enumeration.clnx_version", "CLNX version", base.HEX)
hsb_enumeration_clnx_crc = ProtoField.uint16("hsb-enumeration.clnx_crc", "CLNX CRC", base.HEX)

hsb_enumeration_proto.fields = {
    hsb_enumeration_board_id,
    hsb_enumeration_board_version,
    hsb_enumeration_serial_number,
    hsb_enumeration_cpnx_version,
    hsb_enumeration_cpnx_crc,
    hsb_enumeration_clnx_version,
    hsb_enumeration_clnx_crc,
}

function hsb_enumeration_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "HSB-Enumeration"
    local subtree = tree:add(hsb_enumeration_proto, buffer(), "Holoscan sensor bridge enumeration")
    local offset = 0
    local t_board_id = subtree:add(hsb_enumeration_board_id, buffer(offset, 1))
    offset = offset + 1
    local t_board_version = subtree:add(hsb_enumeration_board_version, buffer(offset, 20))
    offset = offset + 20
    local t_serial_number = subtree:add(hsb_enumeration_serial_number, buffer(offset, 7))
    offset = offset + 7
    local t_cpnx_version = subtree:add_le(hsb_enumeration_cpnx_version, buffer(offset, 2))
    offset = offset + 2
    local t_cpnx_crc = subtree:add_le(hsb_enumeration_cpnx_crc, buffer(offset, 2))
    offset = offset + 2
    local t_clnx_version = subtree:add_le(hsb_enumeration_clnx_version, buffer(offset, 2))
    offset = offset + 2
    local t_clnx_crc = subtree:add_le(hsb_enumeration_clnx_crc, buffer(offset, 2))
    offset = offset + 2

end


--
udp_table = DissectorTable.get("udp.port")
udp_table:add(8192, hsb_proto)
udp_table:add(10001, hsb_enumeration_proto)

