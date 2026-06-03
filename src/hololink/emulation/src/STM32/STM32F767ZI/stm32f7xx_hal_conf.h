/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * See README.md for detailed information.
 */

#ifndef STM32F767ZI_HAL_CONF_H
#define STM32F767ZI_HAL_CONF_H

#ifdef __cplusplus
 extern "C" {
#endif

#define HAL_MODULE_ENABLED
#define HAL_DAC_MODULE_ENABLED
#define HAL_ETH_MODULE_ENABLED
#define HAL_TIM_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED
#define HAL_PCD_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_EXTI_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_I2C_MODULE_ENABLED
#define HAL_SPI_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED

/**
  * @brief Adjust the value of External High Speed oscillator (HSE)
  */
#if !defined  (HSE_VALUE)
  #define HSE_VALUE    ((uint32_t)8000000U) /* in Hz */
#endif

#if !defined  (HSE_STARTUP_TIMEOUT)
  #define HSE_STARTUP_TIMEOUT    ((uint32_t)100U)   /* in ms */
#endif

/**
  * @brief Internal High Speed oscillator (HSI) value.
  */
#if !defined  (HSI_VALUE)
  #define HSI_VALUE    ((uint32_t)16000000U) /* in Hz*/
#endif

/**
  * @brief Internal Low Speed oscillator (LSI) value.
  */
#if !defined  (LSI_VALUE)
 #define LSI_VALUE  ((uint32_t)32000U)       /* in Hz*/
#endif

/**
  * @brief External Low Speed oscillator (LSE) value.
  */
#if !defined  (LSE_VALUE)
 #define LSE_VALUE  ((uint32_t)32768U)    /* in Hz */
#endif

#if !defined  (LSE_STARTUP_TIMEOUT)
  #define LSE_STARTUP_TIMEOUT    ((uint32_t)5000U)   /* in ms */
#endif 

/**
  * @brief External clock source for I2S peripheral
  */
#if !defined  (EXTERNAL_CLOCK_VALUE)
  #define EXTERNAL_CLOCK_VALUE    ((uint32_t)12288000U) /* in Hz*/
#endif

#define  VDD_VALUE                    3300U /* in mv */
#define  TICK_INT_PRIORITY            ((uint32_t)0U)
#define  USE_RTOS                     0U
#define  PREFETCH_ENABLE              0U
#define  ART_ACCELERATOR_ENABLE        0U /* To enable instruction cache and prefetch */

// callback registration
#define  USE_HAL_ETH_REGISTER_CALLBACKS         1U

/* Definition of the Ethernet driver buffers size and count */
#define ETH_RX_BUF_SIZE                1524 /* buffer size for receive 1500 MTU + 14-byte ethernet header + 4 byte VLAN (not yet supported) + 4 byte CRC + 2 bytes for aligning control plane messages */
#define ETH_TX_BUF_SIZE                (ETH_RX_BUF_SIZE - 2) /* ETH_RX_BUF_SIZE less 2 bytes because aligning control plane messages is unneeded */
#define ETH_RXBUFNB                    ((uint32_t)4U)       /* Rx buffer count */
#define ETH_TXBUFNB                    ((uint32_t)4U)       /* Tx buffer count */

/* DP83848_PHY_ADDRESS Address*/
#define DP83848_PHY_ADDRESS
/* PHY Reset delay. Requires 1ms SysTick */
#define PHY_RESET_DELAY                 ((uint32_t)0x000000FFU)
/* PHY Configuration delay */
#define PHY_CONFIG_DELAY                ((uint32_t)0x00000FFFU)

#define PHY_READ_TO                     ((uint32_t)0x0000FFFFU)
#define PHY_WRITE_TO                    ((uint32_t)0x0000FFFFU)

#define PHY_BCR                         ((uint16_t)0x0000U)    /*!< Transceiver Basic Control Register   */
#define PHY_BSR                         ((uint16_t)0x0001U)    /*!< Transceiver Basic Status Register    */

#define PHY_RESET                       ((uint16_t)0x8000U)  /*!< PHY Reset */
#define PHY_LOOPBACK                    ((uint16_t)0x4000U)  /*!< Select loop-back mode */
#define PHY_FULLDUPLEX_100M             ((uint16_t)0x2100U)  /*!< Set the full-duplex mode at 100 Mb/s */
#define PHY_HALFDUPLEX_100M             ((uint16_t)0x2000U)  /*!< Set the half-duplex mode at 100 Mb/s */
#define PHY_FULLDUPLEX_10M              ((uint16_t)0x0100U)  /*!< Set the full-duplex mode at 10 Mb/s  */
#define PHY_HALFDUPLEX_10M              ((uint16_t)0x0000U)  /*!< Set the half-duplex mode at 10 Mb/s  */
#define PHY_AUTONEGOTIATION             ((uint16_t)0x1000U)  /*!< Enable auto-negotiation function     */
#define PHY_RESTART_AUTONEGOTIATION     ((uint16_t)0x0200U)  /*!< Restart auto-negotiation function    */
#define PHY_POWERDOWN                   ((uint16_t)0x0800U)  /*!< Select the power down mode           */
#define PHY_ISOLATE                     ((uint16_t)0x0400U)  /*!< Isolate PHY from MII                 */

#define PHY_AUTONEGO_COMPLETE           ((uint16_t)0x0020U)  /*!< Auto-Negotiation process completed   */
#define PHY_LINKED_STATUS               ((uint16_t)0x0004U)  /*!< Valid link established               */
#define PHY_JABBER_DETECTION            ((uint16_t)0x0002U)  /*!< Jabber condition detected            */

#define PHY_SR                          ((uint16_t))    /*!< PHY status register Offset                      */

#define PHY_SPEED_STATUS                ((uint16_t))  /*!< PHY Speed mask                                  */
#define PHY_DUPLEX_STATUS               ((uint16_t))  /*!< PHY Duplex mask                                 */

/* ################## SPI peripheral configuration ########################## */

/* CRC FEATURE: Use to activate CRC feature inside HAL SPI Driver
* Activated: CRC code is present inside driver
* Deactivated: CRC code cleaned from driver
*/

#define USE_SPI_CRC                     0U

#ifdef HAL_RCC_MODULE_ENABLED
    #include "stm32f7xx_hal_rcc.h"
#endif 

#ifdef HAL_EXTI_MODULE_ENABLED
    #include "stm32f7xx_hal_exti.h"
#endif 

#ifdef HAL_GPIO_MODULE_ENABLED
    #include "stm32f7xx_hal_gpio.h"
#endif 

#ifdef HAL_DMA_MODULE_ENABLED
    #include "stm32f7xx_hal_dma.h"
#endif 

#ifdef HAL_CORTEX_MODULE_ENABLED
    #include "stm32f7xx_hal_cortex.h"
#endif 

#ifdef HAL_ADC_MODULE_ENABLED
    #include "stm32f7xx_hal_adc.h"
#endif 

#ifdef HAL_CAN_MODULE_ENABLED
    #include "stm32f7xx_hal_can.h"
#endif 

#ifdef HAL_CEC_MODULE_ENABLED
    #include "stm32f7xx_hal_cec.h"
#endif

#ifdef HAL_CRC_MODULE_ENABLED
    #include "stm32f7xx_hal_crc.h"
#endif 

#ifdef HAL_CRYP_MODULE_ENABLED
    #include "stm32f7xx_hal_cryp.h"
#endif 

#ifdef HAL_DMA2D_MODULE_ENABLED
    #include "stm32f7xx_hal_dma2d.h"
#endif 

#ifdef HAL_DAC_MODULE_ENABLED
    #include "stm32f7xx_hal_dac.h"
#endif 

#ifdef HAL_DCMI_MODULE_ENABLED
    #include "stm32f7xx_hal_dcmi.h"
#endif 

#ifdef HAL_ETH_MODULE_ENABLED
    #include "stm32f7xx_hal_eth.h"
#endif 

#ifdef HAL_ETH_LEGACY_MODULE_ENABLED
    #include "stm32f7xx_hal_eth_legacy.h"
#endif 

#ifdef HAL_FLASH_MODULE_ENABLED
    #include "stm32f7xx_hal_flash.h"
#endif 

#ifdef HAL_SRAM_MODULE_ENABLED
    #include "stm32f7xx_hal_sram.h"
#endif 

#ifdef HAL_NOR_MODULE_ENABLED
    #include "stm32f7xx_hal_nor.h"
#endif 

#ifdef HAL_NAND_MODULE_ENABLED
    #include "stm32f7xx_hal_nand.h"
#endif 

#ifdef HAL_SDRAM_MODULE_ENABLED
    #include "stm32f7xx_hal_sdram.h"
#endif 

#ifdef HAL_HASH_MODULE_ENABLED
    #include "stm32f7xx_hal_hash.h"
#endif 

#ifdef HAL_I2C_MODULE_ENABLED
    #include "stm32f7xx_hal_i2c.h"
#endif 

#ifdef HAL_I2S_MODULE_ENABLED
    #include "stm32f7xx_hal_i2s.h"
#endif 

#ifdef HAL_IWDG_MODULE_ENABLED
    #include "stm32f7xx_hal_iwdg.h"
#endif 

#ifdef HAL_LPTIM_MODULE_ENABLED
    #include "stm32f7xx_hal_lptim.h"
#endif 

#ifdef HAL_LTDC_MODULE_ENABLED
    #include "stm32f7xx_hal_ltdc.h"
#endif 

#ifdef HAL_PWR_MODULE_ENABLED
    #include "stm32f7xx_hal_pwr.h"
#endif 

#ifdef HAL_QSPI_MODULE_ENABLED
    #include "stm32f7xx_hal_qspi.h"
#endif 

#ifdef HAL_RNG_MODULE_ENABLED
    #include "stm32f7xx_hal_rng.h"
#endif 

#ifdef HAL_RTC_MODULE_ENABLED
    #include "stm32f7xx_hal_rtc.h"
#endif 

#ifdef HAL_SAI_MODULE_ENABLED
    #include "stm32f7xx_hal_sai.h"
#endif 

#ifdef HAL_SD_MODULE_ENABLED
    #include "stm32f7xx_hal_sd.h"
#endif 

#ifdef HAL_MMC_MODULE_ENABLED
    #include "stm32f7xx_hal_mmc.h"
#endif 

#ifdef HAL_SPDIFRX_MODULE_ENABLED
    #include "stm32f7xx_hal_spdifrx.h"
#endif 

#ifdef HAL_SPI_MODULE_ENABLED
    #include "stm32f7xx_hal_spi.h"
#endif 

#ifdef HAL_TIM_MODULE_ENABLED
    #include "stm32f7xx_hal_tim.h"
#endif 

#ifdef HAL_UART_MODULE_ENABLED
    #include "stm32f7xx_hal_uart.h"
#endif 

#ifdef HAL_USART_MODULE_ENABLED
    #include "stm32f7xx_hal_usart.h"
#endif 

#ifdef HAL_IRDA_MODULE_ENABLED
    #include "stm32f7xx_hal_irda.h"
#endif 

#ifdef HAL_SMARTCARD_MODULE_ENABLED
    #include "stm32f7xx_hal_smartcard.h"
#endif 

#ifdef HAL_WWDG_MODULE_ENABLED
    #include "stm32f7xx_hal_wwdg.h"
#endif

#ifdef HAL_PCD_MODULE_ENABLED
    #include "stm32f7xx_hal_pcd.h"
#endif

#ifdef HAL_HCD_MODULE_ENABLED
    #include "stm32f7xx_hal_hcd.h"
#endif

#ifdef HAL_DFSDM_MODULE_ENABLED
    #include "stm32f7xx_hal_dfsdm.h"
#endif 

#ifdef HAL_DSI_MODULE_ENABLED
    #include "stm32f7xx_hal_dsi.h"
#endif 

#ifdef HAL_JPEG_MODULE_ENABLED
    #include "stm32f7xx_hal_jpeg.h"
#endif 

#ifdef HAL_MDIOS_MODULE_ENABLED
    #include "stm32f7xx_hal_mdios.h"
#endif 

#ifdef HAL_SMBUS_MODULE_ENABLED
    #include "stm32f7xx_hal_smbus.h"
#endif 

#ifdef  USE_FULL_ASSERT
  #define assert_param(expr) ((expr) ? (void)0U : assert_failed((uint8_t *)__FILE__, __LINE__))
  void assert_failed(uint8_t* file, uint32_t line);
#else
  #define assert_param(expr) ((void)0U)
#endif 



#ifdef __cplusplus
}
#endif

#endif /* __STM32F7xx_HAL_CONF_H */

